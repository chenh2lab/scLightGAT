#!/usr/bin/env python3
"""Create combined and Overall scLightGAT ablation heatmaps.

The default view shows ``ablation - scLightGAT`` for matched completed seeds.
The full model is used only as the reference and is intentionally not drawn.
The detailed figures cover broad type, the five subtype tasks plus CAF,
immune, and non-immune results. A separate Overall figure is also created for
each family by giving every displayed dataset/task equal weight. Circle color
and the centered number both show the mean score or mean score difference.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


ABLATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_ROOT.parents[1]
DEFAULT_RESULTS_DIR = ABLATION_ROOT / "result"
DEFAULT_SCLIGHTGAT_RESULTS_DIR = PROJECT_ROOT / "result"
DEFAULT_OUT_DIR = ABLATION_ROOT / "plots" / "heatmap"

REFERENCE_METHOD = "scLightGAT"
ABLATION_METHODS = ("No_CDVAE", "No_GAT", "Only_LightGBM")
COMPARTMENTS = (("immune", "Immune"), ("non-immune", "Non-immune"))
METHOD_LABELS = {
    "No_CDVAE": "No CDVAE",
    "No_GAT": "No GAT",
    "Only_LightGBM": "Only LightGBM",
}

METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Acc."),
    ("f1_macro", "Macro-F1"),
    ("f1_weighted", "Weighted-F1"),
)

BROAD_DATASET_ORDER = (
    "GSE115978",
    "GSE123139",
    "GSE153935",
    "GSE166555",
    "Zhengsorted",
    "lung_full",
    "sapiens_full",
)

SUBTYPE_DATASET_ORDER = (
    "B_cell",
    "CD4_T_cell",
    "CD8_T_cell",
    "DC",
    "Plasma_cell",
)

DATASET_LABELS = {
    "B_cell": "B cell",
    "CD4_T_cell": "CD4 T cell",
    "CD8_T_cell": "CD8 T cell",
    "DC": "DC",
    "Plasma_cell": "Plasma cell",
    "CAF": "CAF",
    "lung_full": "Lung",
    "sapiens_full": "Sapiens",
}


@dataclass(frozen=True)
class DatasetSpec:
    """One displayed dataset and its path below each result root."""

    key: str
    parts: tuple[str, ...]
    label: str


def _natural_run_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return 0, int(path.name)
    return 1, path.name


def _read_run_metrics(
    method_dir: Path,
    method: str,
    metric_names: Sequence[str],
    compartment: str | None = None,
) -> dict[int, dict[str, float]]:
    """Read one metric vector per completed seed from a method directory."""
    by_seed: dict[int, dict[str, float]] = {}
    if not method_dir.is_dir():
        return by_seed

    for run_dir in sorted(method_dir.iterdir(), key=_natural_run_key):
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        marker_path = run_dir / "_RUN_COMPLETE.json"
        metrics_path = _metrics_path(run_dir, method, compartment)
        if not marker_path.is_file() or not metrics_path.is_file():
            continue

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(marker.get("status", "complete")).casefold() != "complete":
                continue
            seed = int(marker["seed"])
        except Exception as exc:
            warnings.warn(
                f"Skipping {run_dir}: invalid completion marker ({exc})",
                stacklevel=2,
            )
            continue

        try:
            frame = pd.read_csv(metrics_path)
        except Exception as exc:
            warnings.warn(
                f"Skipping {metrics_path}: could not read CSV ({exc})",
                stacklevel=2,
            )
            continue

        lookup = {str(column).casefold(): column for column in frame.columns}
        if compartment is not None:
            count_source = lookup.get("n_test_cells")
            if count_source is not None:
                counts = pd.to_numeric(frame[count_source], errors="coerce")
                frame = frame.loc[counts > 0]
            if frame.empty:
                continue
        values: dict[str, float] = {}
        for metric in metric_names:
            source = lookup.get(metric.casefold())
            if source is None:
                values[metric] = np.nan
                continue
            numeric = pd.to_numeric(frame[source], errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            values[metric] = (
                float(numeric.mean()) if not numeric.empty else np.nan
            )

        if seed in by_seed:
            warnings.warn(
                f"Duplicate seed {seed} in {method_dir}; keeping the first run",
                stacklevel=2,
            )
            continue
        by_seed[seed] = values

    return by_seed


def _method_dir(root: Path, spec: DatasetSpec, method: str) -> Path:
    return root.joinpath(*spec.parts, method)


def _metrics_path(
    run_dir: Path,
    method: str,
    compartment: str | None = None,
) -> Path:
    if compartment is None:
        return run_dir / f"{method}_metrics.csv"
    return run_dir / compartment / f"{method}_metrics.csv"


def _has_completed_run(
    root: Path,
    spec: DatasetSpec,
    method: str,
    compartment: str | None = None,
) -> bool:
    method_dir = _method_dir(root, spec, method)
    if not method_dir.is_dir():
        return False
    if compartment is not None:
        metrics = _read_run_metrics(
            method_dir,
            method,
            tuple(metric for metric, _ in METRICS),
            compartment,
        )
        return any(
            np.isfinite(value)
            for values in metrics.values()
            for value in values.values()
        )
    return any(
        run_dir.is_dir()
        and run_dir.name.isdigit()
        and (run_dir / "_RUN_COMPLETE.json").is_file()
        and _metrics_path(run_dir, method).is_file()
        for run_dir in method_dir.iterdir()
    )


def _collect_dataset_values(
    spec: DatasetSpec,
    results_dir: Path,
    sclightgat_results_dir: Path,
    *,
    absolute: bool,
    compartment: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Collect paired-seed means for one dataset."""
    metric_names = tuple(metric for metric, _ in METRICS)
    runs = {
        REFERENCE_METHOD: _read_run_metrics(
            _method_dir(
                sclightgat_results_dir,
                spec,
                REFERENCE_METHOD,
            ),
            REFERENCE_METHOD,
            metric_names,
            compartment,
        )
    }
    for method in ABLATION_METHODS:
        runs[method] = _read_run_metrics(
            _method_dir(results_dir, spec, method),
            method,
            metric_names,
            compartment,
        )

    seed_sets = [set(method_runs) for method_runs in runs.values()]
    common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    means = pd.DataFrame(
        np.nan,
        index=pd.Index(ABLATION_METHODS, name="method"),
        columns=pd.Index(metric_names, name="metric"),
        dtype=float,
    )
    if not common_seeds:
        warnings.warn(
            f"{spec.label}: no seed is complete for Full and all ablations",
            stacklevel=2,
        )
        return means, 0

    for method in ABLATION_METHODS:
        for metric in metric_names:
            values = []
            for seed in common_seeds:
                ablation_value = runs[method][seed][metric]
                reference_value = runs[REFERENCE_METHOD][seed][metric]
                if not (
                    np.isfinite(ablation_value)
                    and np.isfinite(reference_value)
                ):
                    continue
                values.append(
                    ablation_value
                    if absolute
                    else ablation_value - reference_value
                )
            if len(values) != len(common_seeds):
                warnings.warn(
                    f"{spec.label}/{method}/{metric}: only {len(values)} of "
                    f"{len(common_seeds)} matched seeds contain finite values; "
                    "leaving the cell missing",
                    stacklevel=2,
                )
                continue
            means.loc[method, metric] = float(np.mean(values))

    return means, len(common_seeds)


def _discover_broad_specs(
    results_dir: Path,
    sclightgat_results_dir: Path,
    compartment: str | None = None,
) -> list[DatasetSpec]:
    excluded = {"Subtype", "CAF", "CAF_test", "CAF_train", "_shared_models"}
    names = set()
    for root in (results_dir, sclightgat_results_dir):
        if not root.is_dir():
            continue
        names.update(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name not in excluded
        )

    ordered = [name for name in BROAD_DATASET_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    specs = []
    for name in ordered:
        spec = DatasetSpec(
            key=name,
            parts=(name,),
            label=DATASET_LABELS.get(name, name.replace("_", " ")),
        )
        if any(
            _has_completed_run(
                sclightgat_results_dir
                if method == REFERENCE_METHOD
                else results_dir,
                spec,
                method,
                compartment,
            )
            for method in (REFERENCE_METHOD, *ABLATION_METHODS)
        ):
            specs.append(spec)
    return specs


def _discover_subtype_caf_specs(
    results_dir: Path,
    sclightgat_results_dir: Path,
) -> list[DatasetSpec]:
    specs = []
    for name in SUBTYPE_DATASET_ORDER:
        spec = DatasetSpec(
            key=name,
            parts=("Subtype", name),
            label=DATASET_LABELS[name],
        )
        if any(
            _has_completed_run(
                sclightgat_results_dir
                if method == REFERENCE_METHOD
                else results_dir,
                spec,
                method,
            )
            for method in (REFERENCE_METHOD, *ABLATION_METHODS)
        ):
            specs.append(spec)

    caf = DatasetSpec(key="CAF", parts=("CAF",), label="CAF")
    if any(
        _has_completed_run(
            sclightgat_results_dir
            if method == REFERENCE_METHOD
            else results_dir,
            caf,
            method,
        )
        for method in (REFERENCE_METHOD, *ABLATION_METHODS)
    ):
        specs.append(caf)
    return specs


def collect_panel(
    specs: Sequence[DatasetSpec],
    results_dir: Path,
    sclightgat_results_dir: Path,
    *,
    absolute: bool,
    compartment: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Combine aligned mean dataset blocks into one wide matrix."""
    mean_blocks = []
    matched_counts = {}
    for spec in specs:
        means, matched = _collect_dataset_values(
            spec,
            results_dir,
            sclightgat_results_dir,
            absolute=absolute,
            compartment=compartment,
        )
        columns = pd.MultiIndex.from_tuples(
            [(spec.key, metric) for metric in means.columns],
            names=("dataset", "metric"),
        )
        means.columns = columns
        mean_blocks.append(means)
        matched_counts[spec.key] = matched

    if not mean_blocks:
        empty = pd.DataFrame(
            index=pd.Index(ABLATION_METHODS, name="method"),
        )
        return empty, matched_counts
    return pd.concat(mean_blocks, axis="columns"), matched_counts


def make_overall_panel(
    matrix: pd.DataFrame,
    specs: Sequence[DatasetSpec],
) -> pd.DataFrame:
    """Return the equal-dataset mean."""
    metric_names = tuple(metric for metric, _ in METRICS)
    means = pd.DataFrame(
        np.nan,
        index=pd.Index(ABLATION_METHODS, name="method"),
        columns=pd.Index(metric_names, name="metric"),
        dtype=float,
    )
    if matrix.empty or not specs:
        return pd.DataFrame(
            index=pd.Index(ABLATION_METHODS, name="method"),
            columns=pd.MultiIndex.from_tuples(
                [],
                names=("dataset", "metric"),
            ),
            dtype=float,
        )

    dataset_keys = [spec.key for spec in specs]
    for metric in metric_names:
        columns = [(dataset, metric) for dataset in dataset_keys]
        values = matrix.loc[:, columns]
        complete = values.notna().all(axis="columns")
        means.loc[complete, metric] = values.loc[complete].mean(axis="columns")

    overall_columns = pd.MultiIndex.from_tuples(
        [("Overall", metric) for metric in metric_names],
        names=("dataset", "metric"),
    )
    means.columns = overall_columns
    return means


BUBBLE_AREA = 1050.0


def _format_score(value: float, *, signed: bool) -> str:
    if abs(value) < 0.005:
        return ".00"
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    if text.startswith("+0."):
        return "+." + text[3:]
    if text.startswith("-0."):
        return "−." + text[3:]
    if text.startswith("0."):
        return text[1:]
    return text.replace("-", "−", 1)


def plot_panel(
    matrix: pd.DataFrame,
    specs: Sequence[DatasetSpec],
    save_path: Path,
    title: str,
    *,
    absolute: bool,
    delta_limit: float,
) -> None:
    """Plot a fixed-size bubble heatmap whose color and label show the mean."""
    if matrix.empty or not specs:
        warnings.warn(f"Skipped {title}: no completed results", stacklevel=2)
        return

    # ``collect_panel`` already stores ablations on rows and dataset/metric on
    # columns.  Full remains a hidden reference and is never drawn.
    values = matrix.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "ablation_score",
        ("#D56D78", "#FAFAFA", "#689BC8"),
    )
    if absolute:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = -float(delta_limit), float(delta_limit)

    n_columns = values.shape[1]
    n_rows = values.shape[0]
    width = max(7.2, 0.78 * n_columns + 3.8)
    height = max(5.4, 0.80 * n_rows + 2.9)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor("white")
    normalizer = Normalize(vmin=vmin, vmax=vmax)

    row_grid, column_grid = np.indices(values.shape)
    finite = np.isfinite(values)
    ax.scatter(
        column_grid[finite],
        row_grid[finite],
        s=BUBBLE_AREA,
        c=values[finite],
        cmap=cmap,
        norm=normalizer,
        marker="o",
        edgecolor="#B0B0B0",
        linewidth=0.45,
        zorder=2,
    )
    ax.set_xlim(-0.5, n_columns - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    metric_labels = {metric: label for metric, label in METRICS}
    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(
        [metric_labels[metric] for _, metric in matrix.columns],
        rotation=42,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [METHOD_LABELS.get(method, method) for method in matrix.index],
        fontsize=11.5,
    )

    ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#E1E1E1", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "–"
            else:
                label = _format_score(value, signed=not absolute)
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="semibold",
                color="#111111",
                zorder=3,
            )

    block_width = len(METRICS)
    for index, spec in enumerate(specs):
        start = index * block_width
        end = start + block_width
        ax.text(
            (start + end - 1) / 2,
            1.045,
            spec.label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
        )
        if index:
            ax.axvline(start - 0.5, color="white", linewidth=5.0)
            ax.axvline(start - 0.5, color="#555555", linewidth=0.7)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=48)
    finite_values = values[np.isfinite(values)]
    has_low_clipping = bool(finite_values.size and finite_values.min() < vmin)
    has_high_clipping = bool(finite_values.size and finite_values.max() > vmax)
    if has_low_clipping and has_high_clipping:
        colorbar_extend = "both"
    elif has_low_clipping:
        colorbar_extend = "min"
    elif has_high_clipping:
        colorbar_extend = "max"
    else:
        colorbar_extend = "neither"
    mappable = ScalarMappable(norm=normalizer, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        pad=0.015,
        fraction=0.022,
        extend=colorbar_extend,
    )
    if absolute:
        colorbar.set_ticks((0.0, 0.5, 1.0))
    else:
        colorbar.set_ticks((-delta_limit, 0.0, delta_limit))
        colorbar.set_ticklabels(
            (f"−{delta_limit:.2f}", "0", f"+{delta_limit:.2f}")
        )
    colorbar.ax.tick_params(labelsize=9.5, colors="#111111")
    colorbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.12, right=0.945, bottom=0.20, top=0.78)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_coverage(
    panel_name: str,
    specs: Sequence[DatasetSpec],
    matched_counts: Mapping[str, int],
) -> None:
    details = ", ".join(
        f"{spec.label}={matched_counts.get(spec.key, 0)}"
        for spec in specs
    )
    print(f"{panel_name} matched seeds: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root containing No_CDVAE, No_GAT and Only_LightGBM results.",
    )
    parser.add_argument(
        "--sclightgat_results_dir",
        type=Path,
        default=DEFAULT_SCLIGHTGAT_RESULTS_DIR,
        help="Root containing hidden full-model reference results.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--delta_limit",
        type=float,
        default=0.10,
        help="Symmetric color limit for ablation-minus-Full score deltas.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Plot absolute ablation scores instead of deltas versus Full.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.delta_limit <= 0:
        raise ValueError("--delta_limit must be positive")

    broad_specs = _discover_broad_specs(
        args.results_dir,
        args.sclightgat_results_dir,
    )
    subtype_caf_specs = _discover_subtype_caf_specs(
        args.results_dir,
        args.sclightgat_results_dir,
    )
    compartment_specs = {
        compartment: _discover_broad_specs(
            args.results_dir,
            args.sclightgat_results_dir,
            compartment,
        )
        for compartment, _ in COMPARTMENTS
    }

    broad, broad_counts = collect_panel(
        broad_specs,
        args.results_dir,
        args.sclightgat_results_dir,
        absolute=args.absolute,
    )
    subtype_caf, subtype_caf_counts = collect_panel(
        subtype_caf_specs,
        args.results_dir,
        args.sclightgat_results_dir,
        absolute=args.absolute,
    )
    compartment_panels = {}
    for compartment, _ in COMPARTMENTS:
        panel, counts = collect_panel(
            compartment_specs[compartment],
            args.results_dir,
            args.sclightgat_results_dir,
            absolute=args.absolute,
            compartment=compartment,
        )
        compartment_panels[compartment] = {
            "panel": panel,
            "counts": counts,
        }
    broad_overall = make_overall_panel(
        broad,
        broad_specs,
    )
    subtype_caf_overall = make_overall_panel(
        subtype_caf,
        subtype_caf_specs,
    )
    for compartment, _ in COMPARTMENTS:
        panel = compartment_panels[compartment]["panel"]
        overall = make_overall_panel(
            panel,
            compartment_specs[compartment],
        )
        compartment_panels[compartment]["overall"] = overall

    _print_coverage("Broad type", broad_specs, broad_counts)
    _print_coverage("Subtype + CAF", subtype_caf_specs, subtype_caf_counts)
    for compartment, scope_label in COMPARTMENTS:
        _print_coverage(
            scope_label,
            compartment_specs[compartment],
            compartment_panels[compartment]["counts"],
        )

    suffix = "Absolute" if args.absolute else "Delta"
    broad_path = args.out_dir / f"Heatmap_Broad_Type_{suffix}.png"
    subtype_caf_path = args.out_dir / f"Heatmap_Subtype_CAF_{suffix}.png"
    broad_overall_path = (
        args.out_dir / f"Heatmap_Broad_Type_Overall_{suffix}.png"
    )
    subtype_caf_overall_path = (
        args.out_dir / f"Heatmap_Subtype_CAF_Overall_{suffix}.png"
    )
    compartment_paths = {}
    for compartment, scope_label in COMPARTMENTS:
        stem = scope_label.replace("-", " ").title().replace(" ", "_")
        compartment_paths[compartment] = {
            "detail": args.out_dir / f"Heatmap_{stem}_{suffix}.png",
            "overall": (
                args.out_dir / f"Heatmap_{stem}_Overall_{suffix}.png"
            ),
        }
    plot_panel(
        broad,
        broad_specs,
        broad_path,
        "Ablation Heatmap · Broad type",
        absolute=args.absolute,
        delta_limit=args.delta_limit,
    )
    plot_panel(
        subtype_caf,
        subtype_caf_specs,
        subtype_caf_path,
        "Ablation Heatmap · Subtype + CAF",
        absolute=args.absolute,
        delta_limit=args.delta_limit,
    )
    overall_spec = (DatasetSpec("Overall", ("Overall",), "Overall"),)
    plot_panel(
        broad_overall,
        overall_spec,
        broad_overall_path,
        "Ablation Heatmap · Broad type Overall",
        absolute=args.absolute,
        delta_limit=args.delta_limit,
    )
    plot_panel(
        subtype_caf_overall,
        overall_spec,
        subtype_caf_overall_path,
        "Ablation Heatmap · Subtype + CAF Overall",
        absolute=args.absolute,
        delta_limit=args.delta_limit,
    )

    for compartment, scope_label in COMPARTMENTS:
        panel_data = compartment_panels[compartment]
        paths = compartment_paths[compartment]
        plot_panel(
            panel_data["panel"],
            compartment_specs[compartment],
            paths["detail"],
            f"Ablation Heatmap · {scope_label}",
            absolute=args.absolute,
            delta_limit=args.delta_limit,
        )
        plot_panel(
            panel_data["overall"],
            overall_spec,
            paths["overall"],
            f"Ablation Heatmap · {scope_label} Overall",
            absolute=args.absolute,
            delta_limit=args.delta_limit,
        )

    if broad_path.is_file():
        print(f"Saved {broad_path}")
    if subtype_caf_path.is_file():
        print(f"Saved {subtype_caf_path}")
    if broad_overall_path.is_file():
        print(f"Saved {broad_overall_path}")
    if subtype_caf_overall_path.is_file():
        print(f"Saved {subtype_caf_overall_path}")
    for compartment, _ in COMPARTMENTS:
        for path in compartment_paths[compartment].values():
            if path.is_file():
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
