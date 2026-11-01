"""Common classification metrics for cell type annotation benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from utils.broad_evaluation_policy import (
    POLICY_VERSION,
    accepted_predictions,
    compare_predictions,
    is_evaluable_ground_truth,
    is_exact_ground_truth,
    normalize_evaluation_label,
    normalize_dataset_name,
)


def _clean_labels(values, missing_value="Unknown"):
    labels = pd.Series(values, dtype="string").str.strip()
    return labels.fillna(missing_value).replace("", missing_value).astype(str)


def _extract_scores(adata, pred_key, score_key, score_classes_key):
    """Return score matrix, class names and an honest score-source label."""
    if score_key in adata.obsm and score_classes_key in adata.uns:
        scores = np.asarray(adata.obsm[score_key], dtype=float)
        classes = np.asarray(adata.uns[score_classes_key]).astype(str)
        if scores.ndim != 2 or scores.shape != (adata.n_obs, len(classes)):
            raise ValueError(
                f"{score_key} must have shape (n_cells, n_score_classes)"
            )
        source = str(adata.uns.get("annotation_score_type", "continuous_scores"))
        return scores, classes, source

    pred = _clean_labels(adata.obs[pred_key])
    classes = np.asarray(sorted(pred.unique()), dtype=str)
    class_to_index = {label: index for index, label in enumerate(classes)}
    scores = np.zeros((adata.n_obs, len(classes)), dtype=float)
    for row, label in enumerate(pred):
        scores[row, class_to_index[label]] = 1.0
    return scores, classes, "hard_predictions"


def _auc_metrics(y_true, scores, score_classes):
    """Calculate one-vs-rest AUCs without requiring identical class sets."""
    class_to_index = {label: index for index, label in enumerate(score_classes)}
    per_class_auc = {}
    supports = {}

    for label in sorted(set(y_true)):
        binary_true = (y_true == label).astype(int)
        supports[label] = int(binary_true.sum())
        if label not in class_to_index or binary_true.min() == binary_true.max():
            per_class_auc[label] = np.nan
            continue
        try:
            per_class_auc[label] = float(
                roc_auc_score(binary_true, scores[:, class_to_index[label]])
            )
        except ValueError:
            per_class_auc[label] = np.nan

    valid = {key: value for key, value in per_class_auc.items() if np.isfinite(value)}
    if not valid:
        return np.nan, np.nan, per_class_auc

    auc_macro = float(np.mean(list(valid.values())))
    total_support = sum(supports[label] for label in valid)
    auc_weighted = float(
        sum(valid[label] * supports[label] for label in valid) / total_support
    )
    return auc_macro, auc_weighted, per_class_auc


def _project_scores_to_source_labels(y_true, scores, score_classes, dataset):
    """Aggregate model scores into the source ground-truth ontology.

    Exact source labels keep their own score column.  A coarse source parent
    receives the sum of every accepted child score (and its own score, if a
    model happens to emit one).  Labels without any score column are omitted
    so :func:`_auc_metrics` reports their AUC as unavailable rather than 0.5.
    """
    normalized_classes = np.asarray(
        [
            normalize_evaluation_label(dataset, label) or str(label)
            for label in score_classes
        ],
        dtype=object,
    )
    projected_scores = []
    projected_classes = []
    for truth in sorted(set(y_true)):
        accepted = set(accepted_predictions(dataset, truth))
        accepted.add(normalize_evaluation_label(dataset, truth) or str(truth))
        indices = np.flatnonzero(np.isin(normalized_classes, list(accepted)))
        if not len(indices):
            continue
        projected_scores.append(np.asarray(scores[:, indices]).sum(axis=1))
        projected_classes.append(str(truth))

    if not projected_scores:
        return np.empty((len(y_true), 0), dtype=float), np.asarray([], dtype=str)
    return (
        np.column_stack(projected_scores),
        np.asarray(projected_classes, dtype=str),
    )


def _averaged_metrics(y_true, y_pred):
    """Return the common macro/micro/weighted precision, recall and F1 set."""
    result = {}
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        result[f"precision_{average}"] = float(precision)
        result[f"recall_{average}"] = float(recall)
        result[f"f1_{average}"] = float(f1)
    return result


def evaluate_predictions(
    adata,
    pred_key,
    celltype_key,
    dataset_name=None,
    score_key="annotation_scores",
    score_classes_key="annotation_score_classes",
):
    """Evaluate predictions in each dataset's source annotation ontology.

    Primary accuracy, precision/recall/F1, AUC, MCC, kappa, per-class metrics
    and the returned confusion matrix all use source-compatible predictions.
    If a source label is coarser than the training ontology, an explicitly
    accepted child prediction is projected back to that source parent before
    metrics are calculated.  Normalized exact-label equality is retained only
    in clearly named ``exact_match_*`` audit fields.  CAF has an exact ontology
    and thus receives no projection or tolerance.
    """
    if celltype_key not in adata.obs:
        raise KeyError(f"Ground-truth column '{celltype_key}' not found in test data")
    if pred_key not in adata.obs:
        raise KeyError(f"Prediction column '{pred_key}' not found")

    dataset = normalize_dataset_name(dataset_name or "")
    raw_true = pd.Series(adata.obs[celltype_key], index=adata.obs_names, dtype="string")
    non_missing = raw_true.notna() & raw_true.str.strip().ne("")
    evaluable = pd.Series(False, index=adata.obs_names)
    if non_missing.any():
        evaluable.loc[non_missing] = raw_true.loc[non_missing].map(
            lambda value: is_evaluable_ground_truth(dataset, value)
        )
    if not evaluable.any():
        raise ValueError(
            "The test data contains no ground-truth labels supported by the "
            f"broad evaluation policy for dataset '{dataset or dataset_name}'"
        )

    comparison = compare_predictions(
        raw_true.loc[evaluable].to_numpy(),
        adata.obs.loc[evaluable, pred_key].to_numpy(),
        dataset,
    )
    y_true = comparison["truth"].astype(str)
    exact_match_pred = comparison["prediction"].astype(str)
    evaluation_pred = comparison["relaxed_prediction"].astype(str)
    scores, score_classes, auc_source = _extract_scores(
        adata, pred_key, score_key, score_classes_key
    )
    scores = scores[np.asarray(evaluable)]
    evaluation_scores, evaluation_score_classes = _project_scores_to_source_labels(
        y_true, scores, score_classes, dataset
    )

    exact_truth = np.asarray(
        [is_exact_ground_truth(dataset, label) for label in y_true], dtype=bool
    )

    # Persist the primary evaluation projection and the exact-match audit
    # decision beside each prediction. Unsupported/missing truth remains out.
    adata.obs["evaluation_included"] = np.asarray(evaluable, dtype=bool)
    for column, values in (
        ("evaluation_prediction", evaluation_pred),
        ("evaluation_correct", comparison["relaxed_correct"]),
        ("exact_match_correct", comparison["strict_correct"]),
        ("compatible_correct", comparison["relaxed_correct"]),
        ("relaxation_rule", comparison["relaxation_rule"]),
        ("relaxation_level", comparison["relaxation_level"]),
    ):
        series = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
        series.loc[evaluable] = values
        adata.obs[column] = series

    labels = np.asarray(sorted(set(y_true) | set(evaluation_pred)), dtype=str)
    primary_averages = _averaged_metrics(y_true, evaluation_pred)
    auc_macro, auc_weighted, per_class_auc = _auc_metrics(
        y_true, evaluation_scores, evaluation_score_classes
    )

    overall = {
        "evaluation_dataset": dataset,
        "evaluation_policy_version": POLICY_VERSION,
        "n_total_cells": int(adata.n_obs),
        "n_missing_ground_truth": int((~non_missing).sum()),
        "n_unsupported_ground_truth": int(
            non_missing.sum() - evaluable.sum()
        ),
        "n_test_cells": int(len(y_true)),
        "evaluation_coverage": float(evaluable.mean()),
        "n_exact_ground_truth": int(exact_truth.sum()),
        "n_coarse_ground_truth": int((~exact_truth).sum()),
        "n_true_classes": int(len(set(y_true))),
        "n_predicted_classes": int(len(set(evaluation_pred))),
        "evaluation_metric_mode": "source_compatible",
        "accuracy": float(accuracy_score(y_true, evaluation_pred)),
        "compatible_accuracy": float(np.mean(comparison["relaxed_correct"])),
        "n_correct": int(np.sum(comparison["relaxed_correct"])),
        "n_compatible_correct": int(np.sum(comparison["relaxed_correct"])),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, evaluation_pred)
        ),
        "auc_ovr_macro": auc_macro,
        "auc_ovr_weighted": auc_weighted,
        "matthews_corrcoef": float(matthews_corrcoef(y_true, evaluation_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, evaluation_pred)),
        "unknown_prediction_rate": float(np.mean(exact_match_pred == "Unknown")),
        "auc_score_source": auc_source,
        "exact_match_accuracy": float(
            accuracy_score(y_true, exact_match_pred)
        ),
        "n_exact_match_correct": int(np.sum(comparison["strict_correct"])),
    }
    overall.update(primary_averages)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, evaluation_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, evaluation_pred, labels=labels)
    per_class_rows = []
    for index, label in enumerate(labels):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        per_class_rows.append({
            "celltype": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": float(specificity),
            "f1": float(f1[index]),
            "auc_ovr": per_class_auc.get(label, np.nan),
            "support": int(support[index]),
        })

    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(cm, index=labels, columns=labels)
    confusion.index.name = "ground_truth"
    confusion.columns.name = "evaluation_prediction"
    return overall, per_class, confusion
