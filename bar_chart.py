"""Create all main-benchmark bar charts from one entry point.

The default score plots include broad type, CAF and Subtype results. A
separate runtime figure reports the average train-plus-predict time across all
seven broad-type datasets. Immune and non-immune compartment plots are produced in
dedicated output subdirectories from the derived metrics stored below each
completed run.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "result"
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "bar_chart"

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
SUBTYPE_BROAD_TYPES = ["B_cell", "CD4_T_cell", "CD8_T_cell", "DC", "Plasma_cell"]
COMPARTMENTS = (("immune", "Immune"), ("non-immune", "Non-immune"))
BROAD_TYPE_DATASETS = (
    "GSE115978",
    "GSE123139",
    "GSE153935",
    "GSE166555",
    "Zhengsorted",
    "lung_full",
    "sapiens_full",
)

METHOD_LABELS = {m: m for m in ALL_METHODS}
COLOR_MAP = plt.get_cmap("Set3")
# Keep the six established method colors stable when adding new baselines.
METHOD_COLOR_INDEX = {
    "Celltypist": 0,
    "CHETAH": 1,
    "Seurat": 2,
    "SingleR": 3,
    "scGPT": 4,
    "scLightGAT": 5,
    "scANVI": 6,
    "scGraphformer": 7,
}
METHOD_COLORS = {
    method: COLOR_MAP(METHOD_COLOR_INDEX[method]) for method in ALL_METHODS
}

METRIC_GROUPS = [
    (
        "Cell-type annotation",
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
    ),
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "Macro-F1",
    "f1_weighted": "Weighted-F1",
}


def _read_values_from_csv(
    csv_path: Path,
    metric_col: str,
    *,
    require_test_cells: bool = False,
) -> List[float]:
    try:
        df = pd.read_csv(csv_path)
        cols_map = {c.upper(): c for c in df.columns}
        target_col = cols_map.get(metric_col.upper())
        if target_col:
            if require_test_cells:
                count_col = cols_map.get("N_TEST_CELLS")
                if count_col:
                    counts = pd.to_numeric(df[count_col], errors="coerce")
                    df = df.loc[counts > 0]
            values = pd.to_numeric(df[target_col], errors="coerce")
            return values.loc[np.isfinite(values)].tolist()
    except Exception:
        pass
    return []


def _metrics_file(
    run_dir: Path,
    method: str,
    compartment: str | None = None,
) -> Path:
    if compartment is None:
        return run_dir / f"{method}_metrics.csv"
    return run_dir / compartment / f"{method}_metrics.csv"


def _get_stats(
    results_dir: Path,
    dataset: str,
    method: str,
    metric: str,
    compartment: str | None = None,
) -> Tuple[float, float]:
    method_dir = results_dir / dataset / method
    if not method_dir.is_dir():
        return np.nan, 0.0

    vals = []
    for run_dir in method_dir.iterdir():
        if (
            not run_dir.is_dir()
            or not run_dir.name.isdigit()
            or not (run_dir / "_RUN_COMPLETE.json").is_file()
        ):
            continue
        metrics_file = _metrics_file(run_dir, method, compartment)
        vals.extend(
            _read_values_from_csv(
                metrics_file,
                metric,
                require_test_cells=compartment is not None,
            )
        )

    if not vals:
        return np.nan, 0.0

    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0


def _has_completed_metrics(
    dataset_dir: Path,
    method: str,
    compartment: str | None = None,
) -> bool:
    method_dir = dataset_dir / method
    if not method_dir.is_dir():
        return False
    for run_dir in method_dir.iterdir():
        metrics_file = _metrics_file(run_dir, method, compartment)
        if (
            run_dir.is_dir()
            and run_dir.name.isdigit()
            and (run_dir / "_RUN_COMPLETE.json").is_file()
            and metrics_file.is_file()
            and any(
                _read_values_from_csv(
                    metrics_file,
                    metric,
                    require_test_cells=compartment is not None,
                )
                for _, metrics in METRIC_GROUPS
                for metric in metrics
            )
        ):
            return True
    return False


def _find_results_dir(
    results_dir: Path,
    sclightgat_results_dir: Path,
    dataset: str,
    method: str,
    compartment: str | None = None,
) -> Path:
    preferred = (
        [sclightgat_results_dir, results_dir]
        if method == "scLightGAT"
        else [results_dir, sclightgat_results_dir]
    )
    checked = set()
    for root in preferred:
        if root in checked:
            continue
        checked.add(root)
        if _has_completed_metrics(root / dataset, method, compartment):
            return root
    return preferred[0]


def _available_methods(
    data_dict: Dict[str, Dict[str, Tuple[float, float]]],
) -> List[str]:
    return [
        method
        for method in ALL_METHODS
        if method in data_dict
        and any(np.isfinite(mean) for mean, _ in data_dict[method].values())
    ]


def _read_runtime_components(
    results_dir: Path,
    dataset: str,
    method: str,
) -> Dict[int, Tuple[float, float]]:
    """Return completed run_id -> (shared train seconds, predict seconds)."""
    method_dir = results_dir / dataset / method
    if not method_dir.is_dir():
        return {}

    values = {}
    for run_dir in sorted(
        method_dir.iterdir(),
        key=lambda path: int(path.name) if path.name.isdigit() else np.inf,
    ):
        if (
            not run_dir.is_dir()
            or not run_dir.name.isdigit()
            or not (run_dir / "_RUN_COMPLETE.json").is_file()
        ):
            continue
        metrics_file = run_dir / f"{method}_metrics.csv"
        try:
            frame = pd.read_csv(metrics_file)
        except Exception:
            continue
        columns = {str(column).casefold(): column for column in frame.columns}
        train_column = columns.get("train_runtime_seconds")
        predict_column = columns.get("predict_runtime_seconds")
        if train_column is None or predict_column is None:
            continue
        train_values = pd.to_numeric(
            frame[train_column], errors="coerce"
        ).to_numpy(dtype=float)
        predict_values = pd.to_numeric(
            frame[predict_column], errors="coerce"
        ).to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values)]
        predict_values = predict_values[np.isfinite(predict_values)]
        if train_values.size == 0 or predict_values.size == 0:
            continue
        values[int(run_dir.name)] = (
            float(np.mean(train_values)),
            float(np.mean(predict_values)),
        )
    return values


def _collect_broad_average_runtime(
    args: argparse.Namespace,
) -> Dict[str, Tuple[float, float, int]]:
    """Calculate mean train-plus-predict runtime across seven broad datasets."""
    stats = {}
    n_datasets = len(BROAD_TYPE_DATASETS)
    for method in ALL_METHODS:
        runtime_by_dataset = {}
        for dataset in BROAD_TYPE_DATASETS:
            root = _find_results_dir(
                args.results_dir,
                args.sclightgat_results_dir,
                dataset,
                method,
            )
            runtime_by_dataset[dataset] = _read_runtime_components(
                root,
                dataset,
                method,
            )

        missing = [
            dataset
            for dataset, run_values in runtime_by_dataset.items()
            if not run_values
        ]
        if missing:
            print(
                f"Skipped broad runtime for {method}: missing completed "
                + ", ".join(missing)
            )
            continue

        common_runs = sorted(
            set.intersection(
                *(set(run_values) for run_values in runtime_by_dataset.values())
            )
        )
        average_runs = []
        for run_id in common_runs:
            components = [
                runtime_by_dataset[dataset][run_id]
                for dataset in BROAD_TYPE_DATASETS
            ]
            train_times = np.asarray(
                [train_time for train_time, _ in components],
                dtype=float,
            )
            if not np.allclose(
                train_times,
                train_times[0],
                rtol=1e-9,
                atol=1e-6,
            ):
                print(
                    f"Skipped broad runtime for {method} run {run_id}: "
                    "shared train time is inconsistent across datasets"
                )
                continue
            predict_total = sum(
                predict_time for _, predict_time in components
            )
            average_runs.append(
                float(train_times[0]) + float(predict_total) / n_datasets
            )

        if not average_runs:
            print(f"Skipped broad runtime for {method}: no complete common run.")
            continue
        stats[method] = (
            float(np.mean(average_runs)),
            (
                float(np.std(average_runs, ddof=1))
                if len(average_runs) > 1
                else 0.0
            ),
            len(average_runs),
        )
    return stats


def _runtime_break_limits(
    means: np.ndarray,
    stds: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], int] | None:
    """Return automatic lower/upper limits when one runtime is an outlier."""
    if means.size < 2:
        return None

    lower_edges = np.maximum(means - stds, 0.0)
    upper_edges = means + stds
    order = np.argsort(upper_edges)
    outlier_index = int(order[-1])
    next_highest = float(upper_edges[order[-2]])

    # Keep a conventional axis unless the tallest bar is clearly separated.
    if next_highest <= 0.0 or lower_edges[outlier_index] < next_highest * 3.0:
        return None

    lower_top = next_highest * 1.12
    outlier_mean = float(means[outlier_index])
    outlier_low = float(lower_edges[outlier_index])
    outlier_high = float(upper_edges[outlier_index])
    upper_padding = max(
        outlier_mean * 0.015,
        (outlier_high - outlier_low) * 2.0,
        1.0,
    )
    upper_bottom = outlier_low - upper_padding
    upper_top = outlier_high + upper_padding
    if upper_bottom <= lower_top:
        return None

    return (0.0, lower_top), (upper_bottom, upper_top), outlier_index


def _draw_horizontal_break(
    ax: plt.Axes,
    y: float,
    *,
    x_center: float,
    width_points: float,
    transform,
    erase_underlay: bool = False,
) -> None:
    """Draw a fixed-size horizontal wave at an axis discontinuity."""
    phase = np.linspace(0.0, 2.0 * np.pi, 41)
    marker = MplPath(
        np.column_stack(
            (
                np.linspace(-1.0, 1.0, phase.size),
                (2.2 / width_points) * np.sin(phase),
            )
        )
    )
    marker_kwargs = {
        "linestyle": "none",
        "marker": marker,
        "markersize": width_points,
        "transform": transform,
        "clip_on": False,
        "zorder": 7,
    }
    if erase_underlay:
        ax.plot(
            [x_center],
            [y],
            color="white",
            markeredgewidth=3.0,
            **marker_kwargs,
        )
    ax.plot(
        [x_center],
        [y],
        color="#222222",
        markeredgewidth=0.9,
        **marker_kwargs,
    )


def _plot_broad_average_runtime(args: argparse.Namespace) -> None:
    """Write one tool-level chart for average broad-type runtime."""
    stats = _collect_broad_average_runtime(args)
    methods = [method for method in ALL_METHODS if method in stats]
    if not methods:
        print("Skipped broad runtime plot: no method has all seven datasets.")
        return

    means = np.asarray([stats[method][0] for method in methods], dtype=float)
    stds = np.asarray([stats[method][1] for method in methods], dtype=float)
    x = np.arange(len(methods))
    break_limits = _runtime_break_limits(means, stds)
    bar_kwargs = {
        "yerr": stds,
        "capsize": 4,
        "color": [METHOD_COLORS[method] for method in methods],
        "edgecolor": "black",
        "linewidth": 0.7,
        "alpha": 0.9,
    }

    if break_limits is None:
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
        ax.bar(x, means, **bar_kwargs)
        axes = (ax,)
        upper = float(np.max(means + stds)) if means.size else 1.0
        ax.set_ylim(0.0, max(upper * 1.08, 1.0))
    else:
        lower_limits, upper_limits, outlier_index = break_limits
        fig, (upper_ax, lower_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(11, 7),
            gridspec_kw={"height_ratios": (1.0, 2.7), "hspace": 0.045},
        )
        for ax in (upper_ax, lower_ax):
            ax.bar(x, means, **bar_kwargs)
        upper_ax.set_ylim(*upper_limits)
        lower_ax.set_ylim(*lower_limits)
        upper_ax.spines["bottom"].set_visible(False)
        lower_ax.spines["top"].set_visible(False)
        upper_ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
        )
        lower_ticks = lower_ax.get_yticks()
        upper_ticks = upper_ax.get_yticks()
        lower_span = lower_limits[1] - lower_limits[0]
        upper_span = upper_limits[1] - upper_limits[0]
        lower_ax.set_yticks(
            lower_ticks[
                (lower_ticks >= lower_limits[0])
                & (lower_ticks < lower_limits[1] - lower_span * 0.03)
            ]
        )
        upper_ax.set_yticks(
            upper_ticks[
                (upper_ticks > upper_limits[0] + upper_span * 0.03)
                & (upper_ticks <= upper_limits[1])
            ]
        )
        axes = (upper_ax, lower_ax)

        # Fixed-point markers keep the upper and lower waves identical even
        # though the two panels have different heights.
        for ax, boundary_y in ((upper_ax, 0.0), (lower_ax, 1.0)):
            _draw_horizontal_break(
                ax,
                boundary_y,
                x_center=0.0,
                width_points=16.0,
                transform=ax.transAxes,
                erase_underlay=True,
            )
            _draw_horizontal_break(
                ax,
                boundary_y,
                x_center=float(x[outlier_index]),
                width_points=68.0,
                transform=ax.get_xaxis_transform(),
                erase_underlay=True,
            )

        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.89)
        upper_position = upper_ax.get_position()
        lower_position = lower_ax.get_position()
        right_spine = upper_ax.spines["right"]
        fig.add_artist(
            Line2D(
                [upper_position.x1, upper_position.x1],
                [lower_position.y1, upper_position.y0],
                transform=fig.transFigure,
                color=right_spine.get_edgecolor(),
                linewidth=right_spine.get_linewidth(),
                zorder=5,
            )
        )

    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(
        [METHOD_LABELS.get(method, method) for method in methods],
        rotation=20,
        ha="right",
        fontsize=11,
    )
    axes[0].set_title(
        "Average Broad-type Runtime",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    for ax in axes:
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    save_path = args.out_dir / "Summary_Broad_Type_Average_Runtime.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 完成: {save_path}")


def _compute_axis_limits(means: np.ndarray, stds: np.ndarray, include_error: bool) -> Tuple[float, float]:
    mask = np.isfinite(means)
    if not np.any(mask):
        return 0.0, 1.0

    valid_means = means[mask]
    valid_stds = stds[mask]

    if include_error:
        lows = valid_means - valid_stds
        highs = valid_means + valid_stds
    else:
        lows = valid_means
        highs = valid_means

    finite_lows = lows[np.isfinite(lows)]
    finite_highs = highs[np.isfinite(highs)]
    if finite_lows.size == 0 or finite_highs.size == 0:
        return 0.0, 1.0

    abs_min = float(np.min(finite_lows))
    abs_max = float(np.max(finite_highs))
    data_range = abs_max - abs_min
    if data_range == 0:
        data_range = max(abs(abs_max) * 0.1, 0.1)

    bottom = max(0.0, abs_min - data_range * 0.1)
    top = abs_max + data_range * 0.1
    if top <= bottom:
        top = bottom + 1.0
    return bottom, top


def plot_dataset_combined(
    dataset_name: str,
    methods: List[str],
    data_dict: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: Path,
    scope_label: str | None = None,
):
    fig, axes = plt.subplots(
        1,
        len(METRIC_GROUPS),
        figsize=(13 * len(METRIC_GROUPS), 7),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    n_methods = len(methods)
    width = 0.8 / n_methods
    handles, labels = [], []

    is_overall = (dataset_name.lower() == "overall")

    for idx, (group_title, metrics) in enumerate(METRIC_GROUPS):
        ax = axes[idx]
        x = np.arange(len(metrics))
        metric_scale = 1.0 / 1024.0 if "Efficiency" in group_title else 1.0

        if "Efficiency" in group_title and len(metrics) == 2:
            ram_metric, vram_metric = metrics
            ax_vram = ax.twinx()
            ram_scale = 1.0 / 1024.0
            vram_scale = 1.0

            ram_means = np.array([data_dict[method][ram_metric][0] * ram_scale for method in methods])
            ram_stds = np.array([data_dict[method][ram_metric][1] * ram_scale for method in methods])
            vram_means = np.array([data_dict[method][vram_metric][0] * vram_scale for method in methods])
            vram_stds = np.array([data_dict[method][vram_metric][1] * vram_scale for method in methods])

            ram_bottom, ram_top = _compute_axis_limits(ram_means, ram_stds, include_error=not is_overall)
            vram_bottom, vram_top = _compute_axis_limits(vram_means, vram_stds, include_error=not is_overall)
            ax.set_ylim(ram_bottom, ram_top)
            ax_vram.set_ylim(vram_bottom, vram_top)

            for i, method in enumerate(methods):
                offset = (i - (n_methods - 1) / 2) * width

                ram_mean, ram_std = data_dict[method][ram_metric]
                vram_mean, vram_std = data_dict[method][vram_metric]
                ram_mean *= ram_scale
                ram_std *= ram_scale
                vram_mean *= vram_scale
                vram_std *= vram_scale

                if np.isfinite(ram_mean):
                    ax.bar(
                        x[0] + offset,
                        ram_mean,
                        width,
                        color=METHOD_COLORS.get(method),
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.9
                    )
                    if not is_overall:
                        ax.errorbar(
                            x[0] + offset,
                            ram_mean,
                            yerr=ram_std,
                            fmt='none',
                            ecolor='black',
                            capsize=2,
                            lw=0.7
                        )

                if np.isfinite(vram_mean):
                    ax_vram.bar(
                        x[1] + offset,
                        vram_mean,
                        width,
                        color=METHOD_COLORS.get(method),
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.9
                    )
                    if not is_overall:
                        ax_vram.errorbar(
                            x[1] + offset,
                            vram_mean,
                            yerr=vram_std,
                            fmt='none',
                            ecolor='black',
                            capsize=2,
                            lw=0.7
                        )

            ax.set_title(group_title, fontsize=14, fontweight='bold', pad=12)
            ax.set_xticks(x)
            display_labels = [METRIC_LABELS.get(m, m) for m in metrics]
            ax.set_xticklabels(display_labels, fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_ylabel("Peak RAM (GB)")
            ax_vram.set_ylabel("Peak VRAM (MB)")
            continue

        mins_incl_error = []
        maxs_incl_error = []

        for m in metrics:
            for method in methods:
                val, std = data_dict[method][m]
                if not np.isnan(val):
                    scaled_val = val * metric_scale
                    scaled_std = std * metric_scale
                    if not is_overall:
                        mins_incl_error.append(scaled_val - scaled_std)
                        maxs_incl_error.append(scaled_val + scaled_std)
                    else:
                        mins_incl_error.append(scaled_val)
                        maxs_incl_error.append(scaled_val)
        if mins_incl_error and maxs_incl_error:
            finite_mins = [v for v in mins_incl_error if np.isfinite(v)]
            finite_maxs = [v for v in maxs_incl_error if np.isfinite(v)]

            if finite_mins and finite_maxs:
                abs_min = min(finite_mins)
                abs_max = max(finite_maxs)

                data_range = abs_max - abs_min
                if data_range == 0:
                    data_range = 0.1

                plot_bottom = abs_min - data_range * 0.1
                if plot_bottom < 0:
                    plot_bottom = 0
                plot_top = abs_max + data_range * 0.1
                plot_top = max(plot_top, abs_max)
                ax.set_ylim(bottom=plot_bottom, top=plot_top)
                y_bottom_for_bars = plot_bottom
            else:
                y_bottom_for_bars = 0
        else:
            y_bottom_for_bars = 0

        if group_title == "Cell-type annotation":
            y_bottom_for_bars = 0.0
            ax.set_ylim(0.0, 1.0)

        for i, method in enumerate(methods):
            means = np.array([data_dict[method][m][0] * metric_scale for m in metrics])
            stds = np.array([data_dict[method][m][1] * metric_scale for m in metrics])

            offset = (i - (n_methods - 1) / 2) * width
            mask = ~np.isnan(means)

            if np.any(mask):
                bar = ax.bar(
                    x[mask] + offset,
                    means[mask] - y_bottom_for_bars,
                    width,
                    bottom=y_bottom_for_bars,
                    color=METHOD_COLORS.get(method),
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.9
                )

                if idx == 0:
                    handles.append(bar)
                    labels.append(METHOD_LABELS.get(method, method))
                if not is_overall:
                    ax.errorbar(
                        x[mask] + offset,
                        means[mask],
                        yerr=stds[mask],
                        fmt='none',
                        ecolor='black',
                        capsize=2,
                        lw=0.7
                    )

        ax.set_title(group_title, fontsize=14, fontweight='bold', pad=12)
        if "Runtime" in group_title:
            ax.set_xticks([])
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.set_xticks(x)
            display_labels = [METRIC_LABELS.get(m, m) for m in metrics]
            ax.set_xticklabels(display_labels, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if "Efficiency" in group_title:
            ax.set_ylabel("Memory (GB)")
        if "Runtime" in group_title:
            ax.set_ylabel("Time (Second)")
        if group_title == "Cell-type annotation":
            ax.set_ylabel("Score")

    scope = f" ({scope_label})" if scope_label else ""
    fig.suptitle(
        f"Benchmark Results{scope}: {dataset_name}",
        fontsize=18,
        fontweight='bold',
        y=1.08,
    )

    fig.legend(handles, 
               labels, 
               loc='center left', 
               bbox_to_anchor=(1.005, 0.5),
               ncol=1, 
               title="Methods", 
               frameon=True, 
               fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_compartment(
    args: argparse.Namespace,
    compartment: str,
    scope_label: str,
) -> None:
    """Plot one broad-type compartment without affecting other scopes."""
    result_roots = {
        root
        for root in (args.results_dir, args.sclightgat_results_dir)
        if root.is_dir()
    }
    excluded = {"caf", "caf_test", "caf_train", "subtype", "_shared_models"}
    datasets = sorted({
        path.name
        for root in result_roots
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name.casefold() not in excluded
        and any(
            _has_completed_metrics(path, method, compartment)
            for method in ALL_METHODS
        )
    })
    if not datasets:
        print(f"Skipped: no dataset has completed {compartment} metrics.")
        return

    results = {
        dataset: {
            method: {
                metric: _get_stats(
                    _find_results_dir(
                        args.results_dir,
                        args.sclightgat_results_dir,
                        dataset,
                        method,
                        compartment,
                    ),
                    dataset,
                    method,
                    metric,
                    compartment,
                )
                for _, metrics in METRIC_GROUPS
                for metric in metrics
            }
            for method in ALL_METHODS
        }
        for dataset in datasets
    }
    datasets = [
        dataset
        for dataset in datasets
        if _available_methods(results[dataset])
    ]
    if not datasets:
        print(f"Skipped: no dataset has valid {compartment} metric values.")
        return

    results["Overall"] = {}
    for method in ALL_METHODS:
        results["Overall"][method] = {}
        for _, group_metrics in METRIC_GROUPS:
            for metric in group_metrics:
                dataset_means = [
                    results[dataset][method][metric][0]
                    for dataset in datasets
                    if np.isfinite(results[dataset][method][metric][0])
                ]
                results["Overall"][method][metric] = (
                    (float(np.mean(dataset_means)), 0.0)
                    if dataset_means
                    else (np.nan, 0.0)
                )

    print(
        f"{scope_label} datasets with completed metrics: "
        + ", ".join(datasets)
    )
    output_dir = args.out_dir / compartment
    for target in datasets + ["Overall"]:
        available_methods = _available_methods(results[target])
        if not available_methods:
            continue
        save_file = output_dir / f"Summary_{target}.png"
        plot_dataset_combined(
            target,
            available_methods,
            results[target],
            save_file,
            scope_label=scope_label,
        )
        print(f"✅ 完成: {save_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--sclightgat_results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    result_roots = {
        root
        for root in (args.results_dir, args.sclightgat_results_dir)
        if root.is_dir()
    }
    if not result_roots:
        print("Skipped: no result directory was found.")
        return

    datasets = sorted({
        path.name
        for root in result_roots
        for path in root.iterdir()
        if path.is_dir()
        and path.name != "Subtype"
        and any(_has_completed_metrics(path, method) for method in ALL_METHODS)
    })
    if not datasets:
        print("Skipped broad plots: no dataset has completed metrics.")

    results = {}

    for ds in datasets:
        results[ds] = {
            method: {
                m: _get_stats(
                    _find_results_dir(
                        args.results_dir,
                        args.sclightgat_results_dir,
                        ds,
                        method,
                    ),
                    ds,
                    method,
                    m,
                )
                for _, ms in METRIC_GROUPS for m in ms
            }
            for method in ALL_METHODS
        }
    datasets = [ds for ds in datasets if _available_methods(results[ds])]

    results["Overall"] = {}
    for method in ALL_METHODS:
        results["Overall"][method] = {}
        for _, group_metrics in METRIC_GROUPS:
            for m in group_metrics:
                ds_means = [
                    results[ds][method][m][0]
                    for ds in datasets
                    if ds.casefold() not in {"caf", "caf_test", "caf_train"}
                    and not np.isnan(results[ds][method][m][0])
                ]
                results["Overall"][method][m] = (
                    (np.mean(ds_means), 0.0) if ds_means else (np.nan, 0.0)
                )
    if datasets:
        print(f"Datasets with completed metrics: {', '.join(datasets)}")
    targets = list(datasets)
    if datasets and _available_methods(results["Overall"]):
        targets.append("Overall")
    for target in targets:
        save_file = args.out_dir / f"Summary_{target}.png"
        plot_dataset_combined(
            target,
            _available_methods(results[target]),
            results[target],
            save_file,
        )
        print(f"✅ 完成: {save_file}")

    _plot_broad_average_runtime(args)

    subtype_results_dir = args.results_dir / "Subtype"
    sclightgat_subtype_results_dir = args.sclightgat_results_dir / "Subtype"
    subtype_datasets = [
        broad_type
        for broad_type in SUBTYPE_BROAD_TYPES
        if any(
            _has_completed_metrics(root / broad_type, method)
            for root in (
                subtype_results_dir,
                sclightgat_subtype_results_dir,
            )
            for method in ALL_METHODS
        )
    ]

    if not subtype_datasets:
        print("No subtype has completed metrics.")
    else:
        print(
            "Subtype broad types with completed metrics: "
            + ", ".join(subtype_datasets)
        )
        for broad_type in subtype_datasets:
            subtype_results = {
                method: {
                    metric: _get_stats(
                        _find_results_dir(
                            subtype_results_dir,
                            sclightgat_subtype_results_dir,
                            broad_type,
                            method,
                        ),
                        broad_type,
                        method,
                        metric,
                    )
                    for _, metrics in METRIC_GROUPS
                    for metric in metrics
                }
                for method in ALL_METHODS
            }
            available_methods = _available_methods(subtype_results)
            if not available_methods:
                continue
            save_file = (
                args.out_dir / "Subtype" / f"Summary_{broad_type}.png"
            )
            plot_dataset_combined(
                broad_type,
                available_methods,
                subtype_results,
                save_file,
            )
            print(f"✅ 完成: {save_file}")

    for compartment, scope_label in COMPARTMENTS:
        _plot_compartment(args, compartment, scope_label)


if __name__ == "__main__":
    main()
