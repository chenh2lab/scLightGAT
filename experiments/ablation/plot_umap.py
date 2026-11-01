"""Plot full-model and ablation predictions on one shared UMAP."""

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_RESULTS_DIR = HERE / "result"
DEFAULT_BASELINE_RESULTS_DIR = PROJECT_ROOT / "result"
DEFAULT_OUT_DIR = HERE / "plots" / "umap"

METHODS = ["scLightGAT", "No_CDVAE", "No_GAT", "Only_LightGBM"]
SUBTYPE_BROAD_TYPES = [
    "B_cell",
    "CD4_T_cell",
    "CD8_T_cell",
    "DC",
    "Plasma_cell",
]
UNKNOWN_LABEL = "Unknown"
DEFAULT_METRIC = "balanced_accuracy"


def _method_root(method, results_dir, baseline_results_dir):
    return baseline_results_dir if method == "scLightGAT" else results_dir


def _metric_mean(path, metric):
    frame = pd.read_csv(path)
    columns = {str(column).casefold(): column for column in frame.columns}
    target = columns.get(metric.casefold())
    if target is None:
        raise KeyError(metric)
    count_column = columns.get("n_test_cells")
    if count_column is not None:
        counts = pd.to_numeric(frame[count_column], errors="coerce")
        frame = frame.loc[counts > 0]
    values = pd.to_numeric(frame[target], errors="coerce")
    values = values.loc[np.isfinite(values)]
    if values.empty:
        raise ValueError(f"No finite {metric} values in {path}")
    return float(values.mean())


def _best_run(
    dataset,
    method,
    results_dir,
    baseline_results_dir,
    metric,
    require_umap=False,
):
    candidates = _scored_runs(
        dataset,
        method,
        results_dir,
        baseline_results_dir,
        metric,
        require_umap=require_umap,
    )
    if not candidates:
        return None, np.nan
    score, _, run_dir = candidates[0]
    return run_dir, score


def _scored_runs(
    dataset,
    method,
    results_dir,
    baseline_results_dir,
    metric,
    require_umap=False,
):
    root = _method_root(method, results_dir, baseline_results_dir)
    method_dir = root / dataset / method
    if not method_dir.is_dir():
        return []
    candidates = []
    for run_dir in method_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        if not (run_dir / "_RUN_COMPLETE.json").is_file():
            continue
        metrics_path = run_dir / f"{method}_metrics.csv"
        predictions_path = run_dir / "predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            continue
        if require_umap and not (run_dir / "umap_coordinates.csv").is_file():
            continue
        try:
            score = _metric_mean(metrics_path, metric)
        except (
            KeyError,
            OSError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            score = np.nan
        candidates.append((score, int(run_dir.name), run_dir))
    return sorted(
        candidates,
        key=lambda item: (
            not np.isfinite(item[0]),
            -item[0] if np.isfinite(item[0]) else 0.0,
            item[1],
        ),
    )


def _normalize_label(value):
    if pd.isna(value):
        return UNKNOWN_LABEL
    label = str(value).strip()
    if not label or label.casefold() in {
        "nan",
        "none",
        "na",
        "unknown",
        "unassigned",
        "unlabeled",
        "unlabelled",
    }:
        return UNKNOWN_LABEL
    return label


def _baseline_frame(run_dir):
    frame = pd.read_csv(
        run_dir / "umap_coordinates.csv",
        dtype={"barcode": str},
    )
    if "barcode" not in frame:
        raise ValueError(f"Missing barcode column in {run_dir}")
    if frame["barcode"].duplicated().any():
        raise ValueError(f"Duplicate barcodes in {run_dir}")
    coordinate_columns = [
        column for column in frame.columns if "umap" in str(column).casefold()
    ][:2]
    if len(coordinate_columns) != 2:
        raise ValueError(f"Missing two UMAP columns in {run_dir}")
    coordinates = frame[coordinate_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = np.isfinite(coordinates.to_numpy()).all(axis=1)
    if not finite.all():
        raise ValueError(f"Non-finite UMAP coordinates in {run_dir}")
    frame = frame.set_index("barcode")
    frame.index = frame.index.astype(str)
    truth_column = "celltype"
    if truth_column not in frame:
        predictions = pd.read_csv(
            run_dir / "predictions.csv",
            dtype={"barcode": str},
        ).set_index("barcode")
        if truth_column not in predictions:
            raise ValueError(f"Missing ground truth in {run_dir}")
        frame[truth_column] = predictions[truth_column].reindex(frame.index)
    frame["ground_truth"] = frame[truth_column].map(_normalize_label)
    frame["umap_x"] = coordinates.iloc[:, 0].to_numpy()
    frame["umap_y"] = coordinates.iloc[:, 1].to_numpy()
    return frame[["umap_x", "umap_y", "ground_truth"]]


def _predictions(run_dir, barcodes):
    frame = pd.read_csv(
        run_dir / "predictions.csv",
        dtype={"barcode": str},
    )
    required = {"barcode", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {run_dir}")
    if frame["barcode"].duplicated().any():
        raise ValueError(f"Duplicate prediction barcodes in {run_dir}")
    values = frame.set_index("barcode")["prediction"].reindex(barcodes)
    return values.map(_normalize_label)


def _discover_datasets(results_dir, baseline_results_dir, metric):
    dataset_names = set()
    for root in (results_dir, baseline_results_dir):
        if not root.is_dir():
            continue
        dataset_names.update(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name.casefold() != "subtype"
        )
    return [
        dataset
        for dataset in sorted(dataset_names)
        if any(
            _best_run(
                dataset,
                method,
                results_dir,
                baseline_results_dir,
                metric,
            )[0]
            is not None
            for method in METHODS
        )
    ]


def _color_map(panel_labels):
    labels = sorted(
        {
            label
            for values in panel_labels.values()
            for label in values.astype(str).unique()
            if label != UNKNOWN_LABEL
        }
    )
    if UNKNOWN_LABEL in {
        label
        for values in panel_labels.values()
        for label in values.astype(str).unique()
    }:
        labels.append(UNKNOWN_LABEL)
    palette = plt.get_cmap("tab20", max(1, len(labels)))
    colors = {label: palette(index) for index, label in enumerate(labels)}
    if UNKNOWN_LABEL in colors:
        colors[UNKNOWN_LABEL] = "#BDBDBD"
    return labels, colors


def _plot_dataset(
    dataset,
    results_dir,
    baseline_results_dir,
    output_dir,
    metric,
    point_size,
):
    runs = {}
    for method in METHODS:
        runs[method], _ = _best_run(
            dataset,
            method,
            results_dir,
            baseline_results_dir,
            metric,
        )

    baseline = None
    anchor_errors = []
    for method in METHODS:
        for _, _, run_dir in _scored_runs(
            dataset,
            method,
            results_dir,
            baseline_results_dir,
            metric,
            require_umap=True,
        ):
            try:
                baseline = _baseline_frame(run_dir)
            except (
                OSError,
                KeyError,
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as error:
                anchor_errors.append(f"{method}/{run_dir.name}: {error}")
                continue
            break
        if baseline is not None:
            break
    if baseline is None:
        detail = f" ({'; '.join(anchor_errors)})" if anchor_errors else ""
        raise RuntimeError(
            f"No completed method run with usable UMAP coordinates for "
            f"{dataset}{detail}"
        )

    panels = {"Ground truth": baseline["ground_truth"]}
    for method in METHODS:
        if runs[method] is None:
            continue
        try:
            panels[method] = _predictions(runs[method], baseline.index)
        except (
            OSError,
            KeyError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            continue

    labels, colors = _color_map(panels)
    panel_order = ["Ground truth"] + [
        method for method in METHODS if method in panels
    ]
    n_columns = min(3, len(panel_order))
    n_rows = int(np.ceil(len(panel_order) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.8 * n_columns, 5.6 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, title in zip(flat_axes, panel_order):
        values = panels[title]
        for label in labels:
            selected = values.to_numpy(dtype=str) == label
            if not selected.any():
                continue
            axis.scatter(
                baseline.loc[selected, "umap_x"],
                baseline.loc[selected, "umap_y"],
                s=point_size,
                c=[colors[label]],
                linewidths=0,
                rasterized=True,
            )
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_aspect("equal", adjustable="datalim")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=colors[label],
            markeredgecolor="none",
            label=label,
        )
        for label in labels
    ]
    unused_axes = list(flat_axes[len(panel_order):])
    if unused_axes:
        legend_axis = unused_axes[0]
        legend_axis.set_axis_off()
        legend_axis.legend(
            handles=handles,
            loc="center",
            title="Cell type",
            frameon=True,
        )
        for axis in unused_axes[1:]:
            figure.delaxes(axis)
        right_margin = 0.98
    else:
        figure.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            title="Cell type",
            frameon=True,
        )
        right_margin = 0.86
    figure.suptitle(
        f"scLightGAT Ablation: {dataset}",
        fontsize=20,
        fontweight="bold",
    )
    top_margin = 0.84 if n_rows == 1 else 0.92
    figure.subplots_adjust(
        right=right_margin,
        top=top_margin,
        wspace=0.04,
        hspace=0.10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Summary_{dataset}.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Completed: {output_path}")


def _plot_or_skip(
    dataset,
    results_dir,
    baseline_results_dir,
    output_dir,
    metric,
    point_size,
):
    try:
        _plot_dataset(
            dataset,
            results_dir,
            baseline_results_dir,
            output_dir,
            metric,
            point_size,
        )
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Skipped {dataset}: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--baseline_results_dir",
        type=Path,
        default=DEFAULT_BASELINE_RESULTS_DIR,
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()

    datasets = args.datasets or _discover_datasets(
        args.results_dir,
        args.baseline_results_dir,
        args.metric,
    )
    if not datasets:
        print("Skipped broad UMAP plots: no complete ablation datasets were found")
    for dataset in datasets:
        _plot_or_skip(
            dataset,
            args.results_dir,
            args.baseline_results_dir,
            args.out_dir,
            args.metric,
            args.point_size,
        )

    subtype_results_dir = args.results_dir / "Subtype"
    subtype_baseline_results_dir = args.baseline_results_dir / "Subtype"
    subtype_datasets = [
        broad_type
        for broad_type in SUBTYPE_BROAD_TYPES
        if any(
            _best_run(
                broad_type,
                method,
                subtype_results_dir,
                subtype_baseline_results_dir,
                args.metric,
            )[0]
            is not None
            for method in METHODS
        )
    ]
    if not subtype_datasets:
        print("No complete subtype ablation datasets were found")
        return

    print("Ablation subtype datasets: " + ", ".join(subtype_datasets))
    for broad_type in subtype_datasets:
        _plot_or_skip(
            broad_type,
            subtype_results_dir,
            subtype_baseline_results_dir,
            args.out_dir / "Subtype",
            args.metric,
            args.point_size,
        )


if __name__ == "__main__":
    main()
