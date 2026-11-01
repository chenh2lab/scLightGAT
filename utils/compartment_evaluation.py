"""Evaluate cell-type annotation separately within each truth compartment.

EDIT THE DEFINITIONS BELOW to change immune/non-immune membership. Labels are
normalized by the existing broad evaluation policy; neither list is inferred
from predictions. Keep this file identical in the scLightGAT and
celltype_annotation projects when changing definitions.

This is NOT an immune-vs-non-immune binary classifier: cell-type labels stay
intact, cells are selected by ground truth, and predictions outside the group
remain errors in its confusion matrix. All metrics (including precision and
specificity) are conditional on the selected truth compartment. Existing
source-compatible/coarse-label rules and unsupported-truth exclusions apply.
Empty compartments produce header-only matrices and NaN quality metrics, not
zero scores. CAF is excluded. Original whole-dataset files are never changed.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.broad_evaluation_policy import (
    POLICY_VERSION,
    SUBTYPE_DATASET_EXACT_LABELS,
    is_evaluable_ground_truth,
    normalize_broad_label,
    normalize_dataset_name,
)
from utils.evaluation import evaluate_predictions


# -------------------- USER-EDITABLE DEFINITIONS --------------------
# Use source/canonical cell-type labels, NOT gene markers or predictions.
# Coarse author labels must be listed too (T.cells, Mono/Mac, Immune, etc.).
IMMUNE_LABELS = {
    "B cells",
    "CD4+T cells",
    "CD8+T cells",
    "DC",
    "Macrophages",
    "Mast cells",
    "Monocytes",
    "NK cells",
    "Plasma cells",
    "pDC",
    "T.cells",
    "Mono/Mac",
    "Immune",
}
NON_IMMUNE_LABELS = {
    "Endothelial cells",
    "Epithelial cells",
    "Fibroblasts",
    "Mesenchymal",
}
SKIP_DATASETS = {
    "CAF",
    "CAF_test",
    "CAF_train",
    *SUBTYPE_DATASET_EXACT_LABELS,
}
# -------------------------------------------------------------------

COMPARTMENTS = ("immune", "non-immune")
SCORE_CACHE_FILE = "annotation_scores.npz"
PER_CLASS_COLUMNS = [
    "celltype", "precision", "recall", "specificity", "f1", "auc_ovr", "support"
]
QUALITY_METRICS = [
    "accuracy", "compatible_accuracy", "balanced_accuracy",
    "auc_ovr_macro", "auc_ovr_weighted", "matthews_corrcoef", "cohen_kappa",
    "unknown_prediction_rate",
    "exact_match_accuracy",
    "specificity_macro", "specificity_weighted",
    *[f"{metric}_{average}" for average in ("macro", "micro", "weighted")
      for metric in ("precision", "recall", "f1")],
]


def compartment_definitions():
    """Return normalized definitions, rejecting ambiguous membership."""
    definitions = {
        "immune": {normalize_broad_label(label) for label in IMMUNE_LABELS},
        "non-immune": {
            normalize_broad_label(label) for label in NON_IMMUNE_LABELS
        },
    }
    if any(None in labels for labels in definitions.values()):
        raise ValueError("Compartment definitions cannot contain missing labels")
    overlap = definitions["immune"] & definitions["non-immune"]
    if overlap:
        raise ValueError(f"Labels occur in both compartments: {sorted(overlap)}")
    return definitions


def compartment_definition_hash():
    """Invalidate derived summaries when the editable definitions change."""
    payload = {
        "scope": "ground_truth_compartment",
        "labels": {key: sorted(value) for key, value in compartment_definitions().items()},
        "skip_datasets": sorted(SKIP_DATASETS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def should_skip_dataset(dataset_name):
    name = normalize_dataset_name(dataset_name).lower()
    return name in {normalize_dataset_name(value).lower() for value in SKIP_DATASETS}


class _PredictionData:
    """The evaluator's small AnnData-compatible interface; no expression copy."""

    def __init__(self, obs, scores=None, classes=None, score_source=None):
        self.obs = obs.copy()
        self.obsm = {}
        self.uns = {}
        if scores is not None:
            scores = np.asarray(scores)
            classes = np.asarray(classes, dtype=str)
            if scores.ndim != 2 or scores.shape != (len(obs), len(classes)):
                raise ValueError("Class scores must have shape (n_cells, n_classes)")
            self.obsm["annotation_scores"] = scores
            self.uns["annotation_score_classes"] = classes
            self.uns["annotation_score_type"] = score_source or "continuous_scores"

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def n_obs(self):
        return len(self.obs)


def _empty_metrics(n_cells):
    metrics = {metric: np.nan for metric in QUALITY_METRICS}
    metrics.update({
        "n_total_cells": int(n_cells),
        "n_missing_ground_truth": 0,
        "n_unsupported_ground_truth": int(n_cells),
        "n_test_cells": 0,
        "evaluation_coverage": 0.0 if n_cells else np.nan,
        "n_exact_ground_truth": 0,
        "n_coarse_ground_truth": 0,
        "n_true_classes": 0,
        "n_predicted_classes": 0,
        "n_correct": 0,
        "n_compatible_correct": 0,
        "n_exact_match_correct": 0,
        "evaluation_metric_mode": "source_compatible",
        "auc_score_source": "unavailable_empty_compartment",
    })
    return metrics


def _specificity_averages(per_class):
    # Match balanced accuracy's true-class scope: omit predicted-only classes.
    valid = per_class["support"].gt(0) & per_class["specificity"].notna()
    values = per_class.loc[valid]
    if values.empty:
        return {"specificity_macro": np.nan, "specificity_weighted": np.nan}
    return {
        "specificity_macro": float(values["specificity"].mean()),
        "specificity_weighted": float(np.average(
            values["specificity"], weights=values["support"]
        )),
    }


def _save_score_cache(adata, pred_key, output_dir):
    """Preserve continuous scores once per run for exact future recalculation."""
    if "annotation_scores" not in adata.obsm:
        return
    classes = adata.uns.get("annotation_score_classes")
    if classes is None:
        return
    scores = np.asarray(adata.obsm["annotation_scores"])
    if scores.ndim != 2 or scores.shape != (adata.n_obs, len(classes)):
        raise ValueError("Invalid annotation_scores shape")
    np.savez_compressed(
        Path(output_dir) / SCORE_CACHE_FILE,
        scores=scores,
        classes=np.asarray(classes, dtype=str),
        barcodes=np.asarray(adata.obs_names, dtype=str),
        predictions=np.asarray(adata.obs[pred_key].astype(str), dtype=str),
        score_source=np.asarray(adata.uns.get("annotation_score_type", "continuous_scores")),
    )


def _plot_confusion(confusion, output_path, title):
    # Lazy import: numerical backfill with --no_plots requires no plotting stack.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if confusion.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No evaluable cells in this compartment", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        # Use exactly the whole-dataset heatmap convention.
        from utils.function import save_confusion_heatmap
        save_confusion_heatmap(confusion, output_path, title)


def write_compartment_results(
    adata, pred_key, celltype_key, output_dir, tool_name, dataset_name,
    metadata=None, save_scores=True, auc_available=True, plot=True,
):
    """Write both subgroup evaluations, reusing the existing metric evaluator.

    ``auc_available=False`` prevents unavailable historical continuous scores
    from being silently replaced with a hard-prediction AUC. No runtime/RAM
    metrics are split or attributed to these compartments.
    """
    if should_skip_dataset(dataset_name) or celltype_key is None:
        return []
    if celltype_key not in adata.obs or pred_key not in adata.obs:
        raise KeyError("Compartment evaluation requires truth and prediction columns")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = compartment_definitions()
    definition_hash = compartment_definition_hash()
    truth = adata.obs[celltype_key].map(normalize_broad_label)
    assigned = truth.isin(set().union(*definitions.values()))
    unassigned = sorted(truth.loc[~assigned & truth.notna()].unique())
    if unassigned:
        warnings.warn(
            f"{dataset_name}: truth labels outside compartment definitions "
            f"are excluded: {unassigned}", UserWarning, stacklevel=2,
        )
    if save_scores:
        _save_score_cache(adata, pred_key, output_dir)
    scores = adata.obsm.get("annotation_scores")
    classes = adata.uns.get("annotation_score_classes")
    metadata = metadata or {}
    rows = []
    for compartment in COMPARTMENTS:
        mask = truth.isin(definitions[compartment]).to_numpy()
        subset = _PredictionData(
            adata.obs.loc[mask],
            scores=np.asarray(scores)[mask] if scores is not None and classes is not None else None,
            classes=classes,
            score_source=adata.uns.get("annotation_score_type"),
        )
        has_evaluable = any(
            is_evaluable_ground_truth(dataset_name, label)
            for label in truth.loc[mask].unique()
        )
        if has_evaluable:
            with warnings.catch_warnings():
                # Cross-compartment errors deliberately create predicted-only
                # labels. This sklearn diagnostic is expected for our scope.
                warnings.filterwarnings(
                    "ignore", message="y_pred contains classes not in y_true",
                    category=UserWarning, module=r"sklearn\.metrics\._classification",
                )
                metrics, per_class, confusion = evaluate_predictions(
                    subset, pred_key, celltype_key, dataset_name=dataset_name
                )
            metrics.update(_specificity_averages(per_class))
            status = "computed"
            if not auc_available:
                metrics["auc_ovr_macro"] = np.nan
                metrics["auc_ovr_weighted"] = np.nan
                metrics["auc_score_source"] = "unavailable_no_class_scores"
                per_class["auc_ovr"] = np.nan
        else:
            metrics = _empty_metrics(subset.n_obs)
            per_class = pd.DataFrame(columns=PER_CLASS_COLUMNS)
            confusion = pd.DataFrame(dtype=int)
            confusion.index.name = "ground_truth"
            status = "empty_compartment" if not subset.n_obs else "no_evaluable_ground_truth"
        metrics.update({
            "evaluation_dataset": normalize_dataset_name(dataset_name),
            "evaluation_policy_version": POLICY_VERSION,
            "evaluation_scope": "ground_truth_compartment",
            "compartment": compartment,
            "compartment_definition_hash": definition_hash,
            "compartment_labels": "|".join(sorted(definitions[compartment])),
            "n_dataset_cells": int(adata.n_obs),
            "n_unassigned_ground_truth": int((~assigned).sum()),
            "unassigned_ground_truth_labels": "|".join(unassigned),
            "tool": tool_name,
            "evaluation_status": status,
        })
        for key in ("run_id", "seed", "dataset", "original_evaluation_policy_version"):
            if key in metadata:
                metrics[key] = metadata[key]
        group_dir = output_dir / compartment
        group_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(group_dir / f"{tool_name}_metrics.csv", index=False)
        per_class.to_csv(group_dir / "per_class_metrics.csv", index=False)
        confusion.to_csv(group_dir / "confusion_matrix.csv")
        if plot:
            _plot_confusion(confusion, group_dir / f"{tool_name}_confusion_matrix.png",
                            f"{tool_name}: {compartment} truth cells")
        rows.append(metrics)
    return rows


def _load_score_cache(path, frame):
    """Align by barcode and reject stale caches; never infer scores from max confidence."""
    with np.load(path, allow_pickle=False) as cache:
        required = {"scores", "classes", "barcodes", "predictions", "score_source"}
        if not required.issubset(cache.files):
            raise ValueError(f"Incomplete class-score cache: {path}")
        cached_index = pd.Index(cache["barcodes"].astype(str))
        if not cached_index.is_unique or not frame.index.is_unique:
            raise ValueError("Class-score alignment requires unique barcodes")
        positions = cached_index.get_indexer(frame.index.astype(str))
        if len(cached_index) != len(frame) or np.any(positions < 0):
            raise ValueError("Class-score cache barcodes do not match predictions.csv")
        cached_pred = cache["predictions"][positions]
        current = frame["prediction"].map(normalize_broad_label).fillna("Unknown")
        cached = pd.Series(cached_pred).map(normalize_broad_label).fillna("Unknown")
        if not np.array_equal(current.to_numpy(), cached.to_numpy()):
            raise ValueError("Class-score cache predictions do not match predictions.csv")
        return cache["scores"][positions], cache["classes"].astype(str), str(cache["score_source"].item())


def backfill_run(
    run_dir, dataset_name=None, tool_name=None, output_dir=None,
    allow_hard_auc=False, plot=True,
):
    """Recompute derived subgroup files from one existing run's raw CSVs."""
    run_dir = Path(run_dir)
    tool_name = tool_name or run_dir.parent.name
    original = pd.read_csv(run_dir / f"{tool_name}_metrics.csv")
    if len(original) != 1:
        raise ValueError(f"Expected one overall metrics row in {run_dir}")
    metadata = original.iloc[0].to_dict()
    dataset_name = dataset_name or metadata.get("evaluation_dataset") or run_dir.parent.parent.name
    if should_skip_dataset(dataset_name):
        return []
    frame = pd.read_csv(run_dir / "predictions.csv", dtype={"barcode": str})
    if not {"barcode", "celltype", "prediction"}.issubset(frame):
        raise ValueError(f"predictions.csv requires barcode, celltype, prediction: {run_dir}")
    frame = frame.set_index("barcode")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate prediction barcodes in {run_dir}")
    cache_path = run_dir / SCORE_CACHE_FILE
    scores, classes, source = (None, None, None)
    if cache_path.is_file():
        scores, classes, source = _load_score_cache(cache_path, frame)
    auc_available = scores is not None or allow_hard_auc or metadata.get("auc_score_source") == "hard_predictions"
    data = _PredictionData(frame, scores=scores, classes=classes, score_source=source)
    metadata["original_evaluation_policy_version"] = metadata.get("evaluation_policy_version", "unknown")
    return write_compartment_results(
        data, "prediction", "celltype", output_dir or run_dir, tool_name, dataset_name,
        metadata=metadata, save_scores=False, auc_available=auc_available, plot=plot,
    )


def compartment_results_current(run_dir, tool_name, plot=True):
    """Derived files never participate in training/resume signatures."""
    run_dir = Path(run_dir)
    for compartment in COMPARTMENTS:
        group_dir = run_dir / compartment
        required = [f"{tool_name}_metrics.csv", "per_class_metrics.csv", "confusion_matrix.csv"]
        if plot:
            required.append(f"{tool_name}_confusion_matrix.png")
        if not all((group_dir / filename).is_file() for filename in required):
            return False
        try:
            frame = pd.read_csv(group_dir / f"{tool_name}_metrics.csv")
            if len(frame) != 1:
                return False
            row = frame.iloc[0]
            if row.get("compartment_definition_hash") != compartment_definition_hash():
                return False
            if row.get("evaluation_policy_version") != POLICY_VERSION:
                return False
        except (OSError, ValueError, pd.errors.ParserError):
            return False
    return True


def refresh_compartment_summaries(tool_root, tool_name):
    """Write each group's summary and mean/std CSVs beside the original files."""
    tool_root = Path(tool_root)
    for compartment in COMPARTMENTS:
        rows = []
        for run_dir in sorted(tool_root.iterdir()):
            path = run_dir / compartment / f"{tool_name}_metrics.csv"
            if not run_dir.name.isdigit() or not path.is_file():
                continue
            if any(child.name.casefold() == "_run_in_progress.json" for child in run_dir.iterdir()):
                continue
            frame = pd.read_csv(path)
            if len(frame) == 1:
                rows.append(frame)
        if not rows:
            continue
        summary = pd.concat(rows, ignore_index=True)
        if "run_id" in summary:
            summary = summary.sort_values("run_id")
        if set(summary["compartment_definition_hash"]) != {compartment_definition_hash()}:
            raise ValueError(f"Mixed compartment definitions in {tool_root}; run backfill again")
        prefix = compartment.replace("-", "_")
        summary.to_csv(tool_root / f"{prefix}_metrics_summary.csv", index=False)
        numeric = summary.select_dtypes(include=[np.number]).columns
        aggregate = summary[numeric].agg(["mean", "std"]).T
        aggregate.index.name = "metric"
        aggregate.to_csv(tool_root / f"{prefix}_metrics_aggregate.csv")
