"""Create one shared 5x2 UMAP comparison figure for each test set.

All panels reuse the UMAP coordinates saved by one completed method run.
scLightGAT is preferred when available; UMAP is never recomputed here.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


COMPARTMENT = "immune"
BEST_METRIC = "balanced_accuracy"
GROUND_TRUTH_COL = "Celltype_training"
SUBTYPE_GROUND_TRUTH_COL = "Celltype_subtraining"
UNKNOWN_LABEL = "Unknown"
UNKNOWN_COLOR = "#BDBDBD"

SUBTYPE_BROAD_TYPES = (
    "B_cell",
    "CD4_T_cell",
    "CD8_T_cell",
    "DC",
    "Plasma_cell",
)

ALL_METHODS = [
    "Celltypist",
    "CHETAH",
    "Seurat",
    "SingleR",
    "scANVI",
    "scGraphformer",
    "scGPT",
    "scLightGAT",
]

PANEL_POSITIONS = {
    "Ground truth": (0, 0),
    "Celltypist": (0, 1),
    "CHETAH": (0, 2),
    "Seurat": (0, 3),
    "SingleR": (0, 4),
    "scANVI": (1, 0),
    "scGraphformer": (1, 1),
    "scGPT": (1, 2),
    "scLightGAT": (1, 3),
    "Legend": (1, 4),
}

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT_CANDIDATES = (
    PROJECT_ROOT / "preprocessing" / "scLightGAT_data",
    PROJECT_ROOT.parent / "preprocessing" / "scLightGAT_data",
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if path.is_dir()),
    DATA_ROOT_CANDIDATES[-1],
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "result"
DEFAULT_SCLIGHTGAT_RESULTS_DIR = DEFAULT_RESULTS_DIR
DEFAULT_SUBTYPE_RESULTS_DIR = DEFAULT_RESULTS_DIR / "Subtype"
DEFAULT_SUBTYPE_SCLIGHTGAT_RESULTS_DIR = (
    DEFAULT_SUBTYPE_RESULTS_DIR
)
DEFAULT_SUBTYPE_TEST_DIR = DATA_ROOT / "Subtype"
DEFAULT_TEST_PATHS = [
    DATA_ROOT / "Independent_testing",
    DATA_ROOT / "caf.data" / "caf_test.h5ad",
]
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "umap"
DEFAULT_SUBTYPE_OUT_DIR = DEFAULT_OUT_DIR / "Subtype"

FIGSIZE = (29, 12)
POINT_SIZE = 4.0
DPI = 300

TITLE_FS = 12
SUPTITLE_FS = 24
LEGEND_TITLE_FS = 14
LEGEND_MARKER_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=Path, nargs="+", default=DEFAULT_TEST_PATHS)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--sclightgat_results_dir",
        type=Path,
        default=DEFAULT_SCLIGHTGAT_RESULTS_DIR,
    )
    parser.add_argument(
        "--subtype_results_dir",
        type=Path,
        default=DEFAULT_SUBTYPE_RESULTS_DIR,
    )
    parser.add_argument(
        "--subtype_sclightgat_results_dir",
        type=Path,
        default=DEFAULT_SUBTYPE_SCLIGHTGAT_RESULTS_DIR,
    )
    parser.add_argument(
        "--subtype_test_dir",
        type=Path,
        default=DEFAULT_SUBTYPE_TEST_DIR,
    )
    parser.add_argument(
        "--subtype_out_dir",
        type=Path,
        default=DEFAULT_SUBTYPE_OUT_DIR,
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--test_celltype", type=str, default=GROUND_TRUTH_COL)
    parser.add_argument("--metric", type=str, default=BEST_METRIC)
    parser.add_argument("--point_size", type=float, default=POINT_SIZE)
    return parser.parse_args()


def _discover_test_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".h5ad":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.h5ad")))

    seen = set()
    unique = []
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _dataset_name(test_file: Path) -> str:
    if test_file.stem.casefold() in {"caf", "caf_test"}:
        return "CAF"
    return test_file.stem


def _method_dir(dataset: str, method: str, args: argparse.Namespace) -> Path:
    preferred_roots = (
        [args.sclightgat_results_dir, args.results_dir]
        if method == "scLightGAT"
        else [args.results_dir, args.sclightgat_results_dir]
    )
    checked = set()
    for root in preferred_roots:
        if root in checked:
            continue
        checked.add(root)
        method_dir = root / dataset / method
        if method_dir.is_dir():
            return method_dir
    return preferred_roots[0] / dataset / method


def _metric_mean(metrics_file: Path, metric: str) -> float:
    df = pd.read_csv(metrics_file)
    columns = {str(c).casefold(): c for c in df.columns}
    metric_col = columns[metric.casefold()]

    count_col = columns.get("n_test_cells")
    if count_col is not None:
        counts = pd.to_numeric(df[count_col], errors="coerce")
        df = df.loc[counts > 0]

    values = pd.to_numeric(df[metric_col], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean())


def _best_run(
    dataset: str,
    method: str,
    args: argparse.Namespace,
    require_umap: bool = False,
) -> Tuple[Optional[Path], float]:
    scored = _scored_runs(dataset, method, args, require_umap=require_umap)
    if not scored:
        return None, np.nan
    score, run = scored[0]
    return run, score


def _scored_runs(
    dataset: str,
    method: str,
    args: argparse.Namespace,
    require_umap: bool = False,
) -> List[Tuple[float, Path]]:
    method_dir = _method_dir(dataset, method, args)
    if not method_dir.is_dir():
        return []

    runs = sorted(
        (p for p in method_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    scored: List[Tuple[float, Path]] = []
    for run in runs:
        if not (run / "_RUN_COMPLETE.json").is_file():
            continue
        if not (run / "predictions.csv").is_file():
            continue
        if require_umap and not any(
            path.is_file()
            for path in (
                run / "umap_coordinates.csv",
                run / COMPARTMENT / "umap_coordinates.csv",
            )
        ):
            continue
        candidates = [
            run / f"{method}_metrics.csv",
            run / COMPARTMENT / f"{method}_metrics.csv",
        ]
        metrics_found = False
        score = np.nan
        for metrics_file in candidates:
            if not metrics_file.is_file():
                continue
            metrics_found = True
            try:
                score = _metric_mean(metrics_file, args.metric)
            except (
                OSError,
                KeyError,
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                continue
            if not np.isfinite(score):
                continue
            break
        if metrics_found:
            scored.append((score, run))

    return sorted(
        scored,
        key=lambda item: (
            not np.isfinite(item[0]),
            -item[0] if np.isfinite(item[0]) else 0.0,
            int(item[1].name),
        ),
    )


def _adata_barcodes(adata: ad.AnnData) -> pd.Index:
    for column in ("barcode", "barcodes", "cell_id", "cellid", "cell"):
        if column in adata.obs.columns:
            return pd.Index(adata.obs[column].astype(str))
    return pd.Index(adata.obs_names.astype(str))


def _normalize_prediction(value: object, method: str) -> str:
    if pd.isna(value):
        return UNKNOWN_LABEL

    label = str(value).strip()
    if not label:
        return UNKNOWN_LABEL

    if method == "CHETAH" and label.casefold().startswith("node"):
        return UNKNOWN_LABEL

    if label.casefold() in {
        "unknown", "unassigned", "unlabeled", "unlabelled", "none", "nan", "na"
    }:
        return UNKNOWN_LABEL

    return label


def _load_predictions(run_dir: Path, adata: ad.AnnData, method: str) -> pd.Series:
    df = pd.read_csv(run_dir / "predictions.csv")
    pred_map = (
        df[["barcode", "prediction"]]
        .drop_duplicates(subset="barcode", keep="first")
        .assign(barcode=lambda x: x["barcode"].astype(str))
        .set_index("barcode")["prediction"]
    )

    aligned = pred_map.reindex(_adata_barcodes(adata))
    aligned.index = adata.obs_names
    return aligned.map(lambda x: _normalize_prediction(x, method))


def _ground_truth(adata: ad.AnnData, column: str) -> pd.Series:
    labels = adata.obs[column].astype(object).copy()
    labels.index = adata.obs_names
    return labels.map(
        lambda x: UNKNOWN_LABEL
        if pd.isna(x) or not str(x).strip()
        else str(x).strip()
    )


def _umap_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "umap_coordinates.csv",
        run_dir / COMPARTMENT / "umap_coordinates.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "UMAP coordinate file was not found. Expected one of:\n"
        f"{searched}"
    )


def _load_umap(run_dir: Path, adata: ad.AnnData) -> np.ndarray:
    umap_file = _umap_file(run_dir)
    df = pd.read_csv(umap_file)

    required = {"barcode", "UMAP1", "UMAP2"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{umap_file} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    coord_map = (
        df[["barcode", "UMAP1", "UMAP2"]]
        .drop_duplicates(subset="barcode", keep="first")
        .assign(barcode=lambda x: x["barcode"].astype(str))
        .set_index("barcode")
    )

    barcodes = _adata_barcodes(adata)
    aligned = coord_map.reindex(barcodes)

    missing_mask = aligned[["UMAP1", "UMAP2"]].isna().any(axis=1)
    if missing_mask.any():
        missing_barcodes = barcodes[missing_mask.to_numpy()].tolist()
        preview = ", ".join(missing_barcodes[:5])
        raise ValueError(
            f"{umap_file} does not contain UMAP coordinates for "
            f"{len(missing_barcodes)} test cells. First missing barcodes: {preview}"
        )

    coords = aligned[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError(f"{umap_file} contains non-finite UMAP coordinates")

    return coords


def _palette(truth: pd.Series, predictions: Dict[str, pd.Series]) -> Dict[str, object]:
    labels = {str(value) for value in truth if str(value) != UNKNOWN_LABEL}
    for series in predictions.values():
        labels.update(str(value) for value in series if str(value) != UNKNOWN_LABEL)

    labels = sorted(labels, key=str.casefold)
    colors: List[object] = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(name)
        colors.extend(cmap(i) for i in range(cmap.N))

    if len(labels) > len(colors):
        cmap = plt.get_cmap("hsv")
        colors.extend(cmap(i / len(labels)) for i in range(len(labels)))

    palette = {label: colors[i] for i, label in enumerate(labels)}
    palette[UNKNOWN_LABEL] = UNKNOWN_COLOR
    return palette


def _robust_limits(
    coords: np.ndarray,
    padding: float = 0.15,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min, x_max = np.percentile(coords[:, 0], [1, 99])
    y_min, y_max = np.percentile(coords[:, 1], [1, 99])

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    max_range = max(x_max - x_min, y_max - y_min)
    half = (max_range / 2) * (1 + padding)
    return (cx - half, cx + half), (cy - half, cy + half)


def _plot_panel(
    ax,
    coords: np.ndarray,
    labels: pd.Series,
    palette: Dict[str, object],
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    point_size: float,
) -> None:
    values = labels.astype(str).to_numpy()
    colors = [palette.get(value, UNKNOWN_COLOR) for value in values]

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.margins(0.02)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _plot_missing_panel(ax, method: str) -> None:
    ax.set_title(method, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.text(
        0.5,
        0.5,
        "No result",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _legend_columns(n: int) -> int:
    if n <= 14:
        return 1
    if n <= 28:
        return 2
    return 3


def _legend_fontsize(n: int) -> float:
    if n <= 12:
        return 12.0
    if n <= 24:
        return 10.5
    return 9.0


def plot_dataset(
    test_file: Path,
    args: argparse.Namespace,
    dataset: Optional[str] = None,
) -> Path:
    dataset = dataset or _dataset_name(test_file)

    adata = ad.read_h5ad(test_file)
    truth = _ground_truth(adata, args.test_celltype)

    coords = None
    anchor_errors = []
    anchor_order = ["scLightGAT"] + [
        method for method in ALL_METHODS if method != "scLightGAT"
    ]
    for method in anchor_order:
        for _, run_dir in _scored_runs(
            dataset,
            method,
            args,
            require_umap=True,
        ):
            try:
                coords = _load_umap(run_dir, adata)
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
        if coords is not None:
            break
    if coords is None:
        detail = f" ({'; '.join(anchor_errors)})" if anchor_errors else ""
        raise FileNotFoundError(
            f"No completed method run with usable UMAP coordinates for "
            f"dataset: {dataset}{detail}"
        )
    xlim, ylim = _robust_limits(coords)

    predictions: Dict[str, pd.Series] = {}
    for method in ALL_METHODS:
        run_dir, _ = _best_run(dataset, method, args)
        if run_dir is None or not (run_dir / "predictions.csv").is_file():
            continue
        try:
            predictions[method] = _load_predictions(run_dir, adata, method)
        except (
            OSError,
            KeyError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            continue

    palette = _palette(truth, predictions)

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE)

    r, c = PANEL_POSITIONS["Ground truth"]
    _plot_panel(
        axes[r, c],
        coords,
        truth,
        palette,
        "Ground truth",
        xlim,
        ylim,
        args.point_size,
    )

    for method in ALL_METHODS:
        r, c = PANEL_POSITIONS[method]
        if method not in predictions:
            _plot_missing_panel(axes[r, c], method)
            continue

        _plot_panel(
            axes[r, c],
            coords,
            predictions[method],
            palette,
            method,
            xlim,
            ylim,
            args.point_size,
        )

    r, c = PANEL_POSITIONS["Legend"]
    legend_ax = axes[r, c]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    legend_labels = [label for label in palette if label != UNKNOWN_LABEL]
    if (
        (truth.astype(str) == UNKNOWN_LABEL).any()
        or any(
            (series.astype(str) == UNKNOWN_LABEL).any()
            for series in predictions.values()
        )
    ):
        legend_labels.append(UNKNOWN_LABEL)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[label],
            markeredgecolor="none",
            markersize=LEGEND_MARKER_SIZE,
            label=label,
        )
        for label in legend_labels
    ]

    legend = legend_ax.legend(
        handles,
        legend_labels,
        loc="center",
        title="Cell types",
        fontsize=_legend_fontsize(len(legend_labels)),
        title_fontsize=LEGEND_TITLE_FS,
        frameon=True,
        ncol=_legend_columns(len(legend_labels)),
        columnspacing=1.0,
        handletextpad=0.6,
        labelspacing=0.45,
        borderpad=0.8,
        markerscale=1.15,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.suptitle(
        f"Dataset: {dataset}",
        fontsize=SUPTITLE_FS,
        fontweight="bold",
        y=0.968,
    )

    plt.subplots_adjust(
        left=0.015,
        right=0.99,
        bottom=0.04,
        top=0.885,
        wspace=0.03,
        hspace=0.10,
    )

    fig.canvas.draw()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset)
    output_file = args.out_dir / f"UMAP_{safe_name}.png"
    fig.savefig(output_file, dpi=DPI, facecolor="white")
    plt.close(fig)
    return output_file


def _plot_or_skip(
    test_file: Path,
    args: argparse.Namespace,
    dataset: Optional[str] = None,
) -> None:
    dataset_name = dataset or _dataset_name(test_file)
    if not test_file.is_file():
        print(f"Skipped {dataset_name}: test data not found: {test_file}")
        return
    try:
        output_file = plot_dataset(test_file, args, dataset=dataset)
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        print(f"Skipped {dataset_name}: {error}")
        return
    print(f"Saved: {output_file}")


def main() -> None:
    args = parse_args()
    test_files = _discover_test_files(args.test_path)
    if not test_files:
        print("Skipped broad UMAP plots: no test .h5ad files were found.")
    for test_file in test_files:
        _plot_or_skip(test_file, args)

    subtype_args = argparse.Namespace(**vars(args))
    subtype_args.results_dir = args.subtype_results_dir
    subtype_args.sclightgat_results_dir = args.subtype_sclightgat_results_dir
    subtype_args.out_dir = args.subtype_out_dir
    subtype_args.test_celltype = SUBTYPE_GROUND_TRUTH_COL
    for broad_type in SUBTYPE_BROAD_TYPES:
        test_file = args.subtype_test_dir / f"{broad_type}_test.h5ad"
        _plot_or_skip(test_file, subtype_args, dataset=broad_type)


if __name__ == "__main__":
    main()
