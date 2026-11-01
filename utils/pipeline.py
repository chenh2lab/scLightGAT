"""Unified train/predict/evaluate pipeline for annotation benchmarks.

Each tool implements only model training and prediction. Loading, repeated
runs, metrics, UMAPs, result files and resource tracking are shared here, in
the same style as ``batch_effect/utils/pipeline.py``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, issparse

from utils.broad_evaluation_policy import POLICY_VERSION
from utils.compartment_evaluation import (
    backfill_run,
    compartment_results_current,
    refresh_compartment_summaries,
    should_skip_dataset,
    write_compartment_results,
)
from utils.dataset import load_dataset, model_input_with_shared_expression
from utils.evaluation import evaluate_predictions
from utils.function import MemoryTracker, compute_and_save_umap, save_confusion_heatmap

SCORE_KEY = "annotation_scores"
SCORE_CLASSES_KEY = "annotation_score_classes"


# ============================================================
# Argument Parsing
# ============================================================

def optional_key(value):
    """Convert CLI None/null/empty values into Python None."""
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in {"", "none", "null"} else value


def create_argument_parser(tool_name, *, include_n_jobs=False):
    """Create the standard CLI shared by every annotation method."""
    parser = argparse.ArgumentParser(description=f"Run {tool_name} cell type annotation")
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--train_celltype", type=optional_key, default="celltype")
    parser.add_argument("--test_celltype", type=optional_key, default="celltype")
    parser.add_argument("--train_batch", type=optional_key, default="batch")
    parser.add_argument("--test_batch", type=optional_key, default="batch")
    parser.add_argument(
        "--run_times", "--n_runs", dest="run_times", type=int, default=1
    )
    if include_n_jobs:
        parser.add_argument(
            "--n_jobs",
            type=int,
            default=12,
            help=(
                "Maximum CPU workers/threads available to the method "
                "(default: 12)"
            ),
        )
    parser.add_argument(
        "--test_batch_map",
        default=None,
        help=(
            "Optional comma-separated DATASET=COLUMN mapping used when "
            "--test_path is a directory; use None for datasets without a "
            "batch column"
        ),
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help=(
            "Optional reproducible seed base. By default a random seed "
            "schedule is generated once, persisted, and shared by methods"
        ),
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Keep per-run trained model files",
    )
    return parser


def seed_python_numpy(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate_seed(run_id, base_seed=None):
    if base_seed is not None:
        return int(base_seed + run_id - 1)
    return random.SystemRandom().randrange(1, 2**31 - 1)


# ============================================================
# Python ↔ R Exchange
# ============================================================

def _rscript_executable():
    """Prefer Rscript from the active Python/Conda environment."""
    env_rscript = Path(sys.executable).with_name("Rscript")
    if env_rscript.is_file():
        return str(env_rscript)
    discovered = shutil.which("Rscript")
    if discovered is None:
        raise FileNotFoundError("Rscript was not found in the active environment or PATH")
    return discovered


def export_adata_for_r(adata, output_dir, layer=None):
    """Export genes × cells Matrix Market data and matching metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if not issparse(matrix):
        matrix = csc_matrix(matrix)
    mmwrite(output_dir / "matrix.mtx", matrix.T)
    pd.DataFrame({"barcode": adata.obs_names.astype(str)}).to_csv(
        output_dir / "barcodes.csv", index=False
    )
    pd.DataFrame({"feature": adata.var_names.astype(str)}).to_csv(
        output_dir / "features.csv", index=False
    )
    adata.obs.to_csv(output_dir / "metadata.csv")


def train_r_reference(
    adata, seed, run_out_path, args, r_script, extra_args=None, layer=None
):
    """Train and serialize an R reference model for one run."""
    model_dir = Path(run_out_path) / "_model"
    exchange_dir = model_dir / "train_exchange"
    export_adata_for_r(adata, exchange_dir, layer=layer)
    command = [
        _rscript_executable(), str(r_script),
        "--task", "train_reference",
        "--input_dir", str(exchange_dir),
        "--save_path", str(model_dir),
        "--label_col", args.train_celltype,
        "--seed", str(seed),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)
    shutil.rmtree(exchange_dir)
    return model_dir / "reference.rds"


def predict_r_reference(
    adata,
    model_path,
    pred_key,
    seed,
    run_out_path,
    args,
    r_script,
    score_type="continuous_scores",
    extra_args=None,
    layer=None,
):
    """Run an R predictor and import labels plus optional class scores."""
    predict_dir = Path(run_out_path) / "_r_predict"
    exchange_dir = predict_dir / "test_exchange"
    export_adata_for_r(adata, exchange_dir, layer=layer)
    command = [
        _rscript_executable(), str(r_script),
        "--task", "predict",
        "--input_dir", str(exchange_dir),
        "--reference_rds", str(model_path),
        "--save_path", str(predict_dir),
        "--label_col", args.train_celltype,
        "--seed", str(seed),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)

    pred_frame = pd.read_csv(predict_dir / "predictions.csv", dtype={"barcode": str})
    if not {"barcode", "prediction"}.issubset(pred_frame.columns):
        raise ValueError("R predictions.csv must contain barcode and prediction columns")
    pred = pred_frame.set_index("barcode")["prediction"]
    adata.obs[pred_key] = pred.reindex(adata.obs_names.astype(str)).fillna("Unknown").values

    score_path = predict_dir / "scores.csv"
    if score_path.exists():
        score_frame = pd.read_csv(score_path, dtype={"barcode": str})
        if "barcode" not in score_frame:
            raise ValueError("R scores.csv must contain a barcode column")
        score_frame = score_frame.set_index("barcode").reindex(adata.obs_names.astype(str))
        adata.obsm[SCORE_KEY] = score_frame.to_numpy(dtype=float)
        adata.uns[SCORE_CLASSES_KEY] = np.asarray(score_frame.columns, dtype=str)
        adata.uns["annotation_score_type"] = score_type

    shutil.rmtree(predict_dir)
    return adata


# ============================================================
# Result Writers
# ============================================================

def _save_predictions(adata, pred_key, test_batch, test_celltype, output_dir):
    frame = pd.DataFrame(index=adata.obs_names)
    frame.index.name = "barcode"
    frame["prediction"] = adata.obs[pred_key].astype(str)
    if test_celltype is not None:
        frame["celltype"] = adata.obs[test_celltype].astype(str)
    if test_batch is not None:
        frame["batch"] = adata.obs[test_batch].astype(str)
    if SCORE_KEY in adata.obsm:
        scores = np.asarray(adata.obsm[SCORE_KEY], dtype=float)
        if scores.shape[1] > 0:
            frame["prediction_score"] = np.nanmax(scores, axis=1)
    for column in (
        "evaluation_included",
        "evaluation_prediction",
        "evaluation_correct",
        "exact_match_correct",
        "compatible_correct",
        "relaxation_level",
        "relaxation_rule",
    ):
        if column in adata.obs:
            frame[column] = adata.obs[column].to_numpy()
    frame.to_csv(Path(output_dir) / "predictions.csv")


def _write_run_results(
    adata, metrics, per_class, confusion, tool_name, pred_key, args, seed,
    run_out_path,
):
    run_out_path = Path(run_out_path)
    pd.DataFrame([metrics]).to_csv(
        run_out_path / f"{tool_name}_metrics.csv", index=False
    )
    _save_predictions(
        adata, pred_key, args.test_batch, args.test_celltype, run_out_path
    )

    if per_class is not None and confusion is not None:
        per_class.to_csv(run_out_path / "per_class_metrics.csv", index=False)
        confusion.to_csv(run_out_path / "confusion_matrix.csv")
        save_confusion_heatmap(
            confusion,
            run_out_path / f"{tool_name}_confusion_matrix.png",
            f"{tool_name} prediction",
        )
        write_compartment_results(
            adata, pred_key, args.test_celltype, run_out_path, tool_name,
            metrics["evaluation_dataset"], metadata=metrics,
        )

    compute_and_save_umap(
        adata,
        pred_key,
        args.test_batch,
        args.test_celltype,
        seed,
        run_out_path,
        tool_name,
    )


# ============================================================
# Main Pipeline
# ============================================================

_RUN_COMPLETE_FILE = "_RUN_COMPLETE.json"
_RUN_IN_PROGRESS_FILE = "_RUN_IN_PROGRESS.json"
_MODEL_COMPLETE_FILE = "_MODEL_COMPLETE.json"


def _file_identity(path):
    """Identify data by resolved path and metadata, not pathname alone."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


@lru_cache(maxsize=64)
def _file_sha256(path_string, size, mtime_ns):
    """Hash a stable file snapshot once per process."""
    del size, mtime_ns  # Included in the cache key to invalidate changed files.
    digest = hashlib.sha256()
    with open(path_string, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_file_identity(path):
    """Identify a directory member by portable content, not its location."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "name": resolved.name,
        "size": int(stat.st_size),
        "sha256": _file_sha256(
            str(resolved), int(stat.st_size), int(stat.st_mtime_ns)
        ),
    }


def _json_signature_value(value):
    """Normalize argparse values into a stable JSON-compatible structure."""
    if isinstance(value, Path):
        resolved = value.resolve()
        if resolved.is_file():
            return {"kind": "file", **_file_identity(resolved)}
        if resolved.is_dir():
            files = [
                _directory_file_identity(path)
                for path in sorted(resolved.iterdir())
                if path.is_file()
            ]
            return {
                "kind": "directory",
                "files": files,
            }
        return str(resolved)
    if isinstance(value, dict):
        return {
            str(key): _json_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_signature_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _training_signature(args):
    """Capture every train-affecting CLI value while excluding query/output state."""
    excluded = {
        "base_seed", "run_times", "save_models", "save_path",
        "test_batch", "test_batch_map", "test_celltype", "test_path",
    }
    return {
        key: _json_signature_value(value)
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _training_signatures_match(recorded, expected):
    """Accept obsolete optional fields only when they were explicitly unset."""
    if not isinstance(recorded, dict):
        return False
    normalized = {
        key: value
        for key, value in recorded.items()
        if key in expected or value is not None
    }
    return normalized == expected


def _write_json_atomic(path, payload):
    """Write a small state marker last so interrupted runs stay incomplete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _parse_test_batch_map(value):
    """Parse DATASET=COLUMN pairs supplied by the benchmark launcher."""
    if optional_key(value) is None:
        return {}
    mapping = {}
    for item in str(value).split(","):
        if "=" not in item:
            raise ValueError(
                "--test_batch_map entries must use DATASET=COLUMN syntax"
            )
        dataset, column = (part.strip() for part in item.split("=", 1))
        if not dataset:
            raise ValueError("--test_batch_map contains an empty dataset name")
        mapping[Path(dataset).stem] = optional_key(column)
    return mapping


def _resolve_test_specs(args):
    """Return test path, output root and batch key for each query dataset."""
    test_path = Path(args.test_path)
    if test_path.is_file():
        return [{
            "name": test_path.stem,
            "path": test_path,
            "save_path": Path(args.save_path),
            "test_batch": args.test_batch,
        }]
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test path does not exist: {test_path}")

    test_files = sorted(test_path.glob("*.h5ad"))
    if not test_files:
        raise FileNotFoundError(f"No .h5ad files found in test directory: {test_path}")
    batch_map = _parse_test_batch_map(getattr(args, "test_batch_map", None))
    unknown = sorted(set(batch_map) - {path.stem for path in test_files})
    if unknown:
        raise ValueError(
            "--test_batch_map names files not present in --test_path: "
            + ", ".join(unknown)
        )
    return [
        {
            "name": path.stem,
            "path": path,
            "save_path": Path(args.save_path) / path.stem,
            "test_batch": batch_map.get(path.stem, args.test_batch),
        }
        for path in test_files
    ]


def _seed_for_run(args, tool_name, run_id):
    """Return the persisted shared seed, honoring explicit legacy overrides."""
    identity = {
        "run_id": run_id,
        "train_data": _file_identity(args.train_path),
        "train_celltype": args.train_celltype,
    }

    # A tool-specific state is reserved for importing an already-completed run
    # whose historical seed differs from the shared schedule.  New schedules
    # continue to use the shared state below; this branch never creates an
    # override implicitly.
    if args.base_seed is None:
        override_path = (
            Path(args.save_path) / ".benchmark_state" / "seeds_by_tool"
            / tool_name / f"run_{run_id}.json"
        )
        override = _read_json(override_path)
        override_identity = {**identity, "tool": tool_name}
        if override is not None and all(
            override.get(key) == value
            for key, value in override_identity.items()
        ):
            try:
                return int(override["seed"])
            except (KeyError, TypeError, ValueError):
                pass

    state_path = (
        Path(args.save_path) / ".benchmark_state" / "seeds"
        / f"run_{run_id}.json"
    )
    previous = _read_json(state_path)
    if args.base_seed is None:
        if previous is not None and all(
            previous.get(key) == value for key, value in identity.items()
        ):
            try:
                return int(previous["seed"])
            except (KeyError, TypeError, ValueError):
                pass
        seed = generate_seed(run_id, None)
    else:
        seed = generate_seed(run_id, args.base_seed)

    _write_json_atomic(state_path, {**identity, "seed": seed})
    return seed


def _required_result_files(tool_name, run_out_path, test_args):
    files = [
        f"{tool_name}_metrics.csv",
        "predictions.csv",
        "umap_coordinates.csv",
        f"{tool_name}_umap_prediction.png",
    ]
    if test_args.test_celltype is not None:
        files.extend([
            "per_class_metrics.csv",
            "confusion_matrix.csv",
            f"{tool_name}_confusion_matrix.png",
            f"{tool_name}_umap_celltype.png",
            f"{tool_name}_umap_comparison.png",
        ])
    if test_args.test_batch is not None:
        files.append(f"{tool_name}_umap_batch.png")
    return [Path(run_out_path) / filename for filename in files]


def _run_marker_payload(status, tool_name, run_id, seed, args, test_path):
    return {
        "status": status,
        "tool": tool_name,
        "run_id": int(run_id),
        "seed": int(seed),
        "train_data": _file_identity(args.train_path),
        "test_data": _file_identity(test_path),
        "train_celltype": args.train_celltype,
        "test_celltype": args.test_celltype,
        "train_batch": args.train_batch,
        "test_batch": args.test_batch,
        "evaluation_policy_version": POLICY_VERSION,
        "training_signature": _training_signature(args),
    }


def _run_is_complete(tool_name, run_id, seed, run_out_path, args, test_path):
    """Accept only protocol-matched artifacts with an atomic completion marker."""
    run_out_path = Path(run_out_path)
    if (run_out_path / _RUN_IN_PROGRESS_FILE).exists():
        return False
    marker = _read_json(run_out_path / _RUN_COMPLETE_FILE)
    if marker is None:
        return False
    expected = _run_marker_payload(
        "complete", tool_name, run_id, seed, args, test_path
    )
    if not all(
        _training_signatures_match(marker.get(key), value)
        if key == "training_signature"
        else marker.get(key) == value
        for key, value in expected.items()
    ):
        return False

    required = _required_result_files(tool_name, run_out_path, args)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        metrics = pd.read_csv(run_out_path / f"{tool_name}_metrics.csv")
        if len(metrics) != 1 or int(metrics.iloc[0]["seed"]) != int(seed):
            return False
        if int(metrics.iloc[0]["run_id"]) != int(run_id):
            return False
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError):
        return False
    return True


def _model_marker_payload(tool_name, seed, args, train_runtime, memory):
    return {
        "status": "complete",
        "tool": tool_name,
        "seed": int(seed),
        "train_data": _file_identity(args.train_path),
        "training_signature": _training_signature(args),
        "train_runtime_seconds": float(train_runtime),
        "peak_ram_mb": float(memory["peak_ram_mb"]),
        "peak_vram_mb": float(memory["peak_vram_mb"]),
    }


def _matching_model_marker(shared_run_path, tool_name, seed, args):
    marker = _read_json(Path(shared_run_path) / _MODEL_COMPLETE_FILE)
    if marker is None:
        return None
    expected = {
        "status": "complete",
        "tool": tool_name,
        "seed": int(seed),
        "train_data": _file_identity(args.train_path),
        "training_signature": _training_signature(args),
    }
    matches = all(
        _training_signatures_match(marker.get(key), value)
        if key == "training_signature"
        else marker.get(key) == value
        for key, value in expected.items()
    )
    return marker if matches else None


def _load_completed_model(
    shared_run_path, tool_name, seed, args, load_model_fn,
    model_is_complete_fn,
):
    marker = _matching_model_marker(shared_run_path, tool_name, seed, args)
    if marker is None:
        return None, None
    if model_is_complete_fn is not None and not model_is_complete_fn(
        shared_run_path, seed, args
    ):
        return None, None
    if load_model_fn is not None:
        return load_model_fn(shared_run_path, seed, args), marker

    # All R-backed standard methods use this common persistent handle.
    reference_rds = Path(shared_run_path) / "_model" / "reference.rds"
    if reference_rds.is_file() and reference_rds.stat().st_size > 0:
        return reference_rds, marker
    return None, None


def _refresh_metric_summaries(test_specs, tool_name, run_times):
    for spec in test_specs:
        tool_root = spec["save_path"] / tool_name
        rows = []
        for run_id in range(1, run_times + 1):
            metrics_path = tool_root / str(run_id) / f"{tool_name}_metrics.csv"
            if metrics_path.is_file():
                try:
                    frame = pd.read_csv(metrics_path)
                except (OSError, pd.errors.ParserError):
                    continue
                if len(frame) == 1:
                    rows.append(frame)
        if not rows:
            continue
        summary = pd.concat(rows, ignore_index=True).sort_values("run_id")
        summary.to_csv(tool_root / "metrics_summary.csv", index=False)
        numeric = summary.select_dtypes(include=[np.number]).columns
        aggregate = summary[numeric].agg(["mean", "std"]).T
        aggregate.index.name = "metric"
        aggregate.to_csv(tool_root / "metrics_aggregate.csv")
        refresh_compartment_summaries(tool_root, tool_name)


def _cleanup_model_artifacts(shared_run_path, multi_test):
    shared_run_path = Path(shared_run_path)
    if multi_test:
        if shared_run_path.exists():
            shutil.rmtree(shared_run_path)
        return
    model_dir = shared_run_path / "_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    marker_path = shared_run_path / _MODEL_COMPLETE_FILE
    if marker_path.exists():
        marker_path.unlink()


def _validate_key(adata, key, dataset_name, role):
    if key is not None and key not in adata.obs:
        raise KeyError(f"{dataset_name} {role} column '{key}' not found")


def run_annotation_pipeline(
    tool_name, pred_key, train_fn, predict_fn, args,
    load_model_fn=None, model_is_complete_fn=None,
    expression_layers=None, share_train_expression=False,
    keep_var_metadata=True,
):
    """Train once per seed, then predict one file or every file in a directory.

    ``load_model_fn`` and ``model_is_complete_fn`` are optional resume hooks.
    R-backed methods need no hook because their common ``reference.rds`` model
    handle is detected automatically.
    """
    if args.run_times < 1:
        raise ValueError("--n_runs/--run_times must be at least 1")
    if hasattr(args, "n_jobs") and args.n_jobs < 1:
        raise ValueError("--n_jobs must be at least 1")

    for name in ["train_celltype", "test_celltype", "train_batch", "test_batch"]:
        setattr(args, name, optional_key(getattr(args, name, None)))

    args.train_path = Path(args.train_path)
    args.test_path = Path(args.test_path)
    args.save_path = Path(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)
    if args.train_celltype is None:
        raise ValueError(
            "--train_celltype=None means no training labels, but all methods in "
            "this supervised benchmark require labelled reference cells"
        )

    test_specs = _resolve_test_specs(args)
    multi_test = len(test_specs) > 1
    for spec in test_specs:
        test_args = copy.copy(args)
        test_args.test_path = spec["path"]
        test_args.save_path = spec["save_path"]
        test_args.test_batch = spec["test_batch"]
        spec["args"] = test_args
    print(
        f"Test datasets: {len(test_specs)} | reference will be trained once "
        "per run/seed"
    )
    for spec in test_specs:
        print(
            f"  {spec['name']}: {spec['path']} | "
            f"test_batch={spec['test_batch']}"
        )
    if args.test_celltype is None:
        print("[evaluation] test_celltype=None: classification metrics are skipped")

    seeds = {
        run_id: _seed_for_run(args, tool_name, run_id)
        for run_id in range(1, args.run_times + 1)
    }
    train_raw = None
    for run_id in range(1, args.run_times + 1):
        seed = seeds[run_id]
        seed_python_numpy(seed)

        pending = []
        for spec in test_specs:
            run_out_path = spec["save_path"] / tool_name / str(run_id)
            if _run_is_complete(
                tool_name, run_id, seed, run_out_path, spec["args"], spec["path"]
            ):
                # Supplement completed runs without invalidating model/resume state.
                if (
                    spec["args"].test_celltype is not None
                    and not should_skip_dataset(spec["name"])
                    and not compartment_results_current(run_out_path, tool_name)
                ):
                    backfill_run(run_out_path, spec["name"], tool_name)
                print(
                    f"[resume] {tool_name} run {run_id} / {spec['name']} "
                    "is complete; skipping"
                )
            else:
                pending.append(spec)

        if not pending:
            if not args.save_models:
                if multi_test:
                    completed_model_path = (
                        args.save_path / "_shared_models" / tool_name / str(run_id)
                    )
                else:
                    completed_model_path = (
                        test_specs[0]["save_path"] / tool_name / str(run_id)
                    )
                _cleanup_model_artifacts(completed_model_path, multi_test)
            continue

        print(f"\n{'=' * 60}")
        print(f"{tool_name} run {run_id}/{args.run_times} | seed={seed}")
        print("Pending tests: " + ", ".join(spec["name"] for spec in pending))
        print(f"{'=' * 60}\n")

        if multi_test:
            shared_run_path = (
                args.save_path / "_shared_models" / tool_name / str(run_id)
            )
        else:
            shared_run_path = pending[0]["save_path"] / tool_name / str(run_id)
        shared_run_path.mkdir(parents=True, exist_ok=True)

        model_handle = None
        model_marker = None
        try:
            model_handle, model_marker = _load_completed_model(
                shared_run_path, tool_name, seed, args,
                load_model_fn, model_is_complete_fn,
            )
        except Exception as error:
            print(f"[resume] saved model could not be loaded ({error}); retraining")

        if model_handle is not None:
            print(f"[resume] loading completed {tool_name} model for seed {seed}")
            train_runtime = float(model_marker["train_runtime_seconds"])
            train_memory = {
                "peak_ram_mb": float(model_marker["peak_ram_mb"]),
                "peak_vram_mb": float(model_marker["peak_vram_mb"]),
            }
        else:
            marker_path = shared_run_path / _MODEL_COMPLETE_FILE
            if marker_path.exists():
                marker_path.unlink()
            if train_raw is None:
                train_raw = load_dataset(
                    args.train_path,
                    obs_columns=(args.train_celltype, args.train_batch),
                    layers=expression_layers,
                    keep_var_metadata=keep_var_metadata,
                )
                _validate_key(
                    train_raw, args.train_celltype, "Training", "cell-type"
                )
                _validate_key(train_raw, args.train_batch, "Training", "batch")
                valid_train = train_raw.obs[args.train_celltype].notna()
                valid_train &= (
                    train_raw.obs[args.train_celltype]
                    .astype(str).str.strip().ne("")
                )
                if not valid_train.all():
                    print(
                        f"[train] dropping {int((~valid_train).sum())} cells "
                        "with missing labels"
                    )
                    train_raw = train_raw[valid_train].copy()
                if train_raw.obs[args.train_celltype].nunique() < 2:
                    raise ValueError(
                        "Training data must contain at least two cell types"
                    )
                print(
                    f"Train: {train_raw.n_obs} cells × {train_raw.n_vars} genes\n"
                    f"Train celltype: {args.train_celltype} | "
                    f"Train batch: {args.train_batch}"
                )

            train_tracker = MemoryTracker()
            train_tracker.start()
            train_start = time.time()
            try:
                train_columns = [args.train_celltype]
                if (
                    args.train_batch is not None
                    and args.train_batch not in train_columns
                ):
                    train_columns.append(args.train_batch)
                if share_train_expression:
                    train_input = model_input_with_shared_expression(
                        train_raw, train_columns
                    )
                else:
                    train_input = train_raw.copy()
                    train_input.obs = train_input.obs[train_columns].copy()
                model_handle = train_fn(
                    train_input, seed, shared_run_path, args
                )
                del train_input
                gc.collect()
            except Exception:
                train_tracker.stop()
                raise
            train_runtime = time.time() - train_start
            train_memory = train_tracker.stop()
            model_marker = _model_marker_payload(
                tool_name, seed, args, train_runtime, train_memory
            )
            _write_json_atomic(
                shared_run_path / _MODEL_COMPLETE_FILE, model_marker
            )

        for spec in pending:
            test_args = spec["args"]
            run_out_path = spec["save_path"] / tool_name / str(run_id)
            run_out_path.mkdir(parents=True, exist_ok=True)
            complete_marker_path = run_out_path / _RUN_COMPLETE_FILE
            if complete_marker_path.exists():
                complete_marker_path.unlink()
            progress_marker_path = run_out_path / _RUN_IN_PROGRESS_FILE
            _write_json_atomic(
                progress_marker_path,
                _run_marker_payload(
                    "in_progress", tool_name, run_id, seed,
                    test_args, spec["path"],
                ),
            )
            test_raw = load_dataset(
                spec["path"],
                obs_columns=(test_args.test_celltype, test_args.test_batch),
                layers=expression_layers,
                keep_var_metadata=keep_var_metadata,
            )
            _validate_key(
                test_raw, test_args.test_celltype,
                f"Test ({spec['name']})", "cell-type",
            )
            _validate_key(
                test_raw, test_args.test_batch,
                f"Test ({spec['name']})", "batch",
            )
            print(
                f"\n[{spec['name']}] Test: {test_raw.n_obs} cells × "
                f"{test_raw.n_vars} genes | batch={test_args.test_batch}"
            )

            # Prediction receives expression plus the explicitly configured
            # batch column only. Ground truth and all other source annotation
            # metadata are restored after inference, so no method can use them
            # accidentally while fitting or predicting.
            test_metadata = test_raw.obs.copy()
            if test_args.test_batch is None:
                test_raw.obs = pd.DataFrame(index=test_raw.obs_names)
            else:
                test_raw.obs = test_metadata[[test_args.test_batch]].copy()

            predict_tracker = MemoryTracker()
            predict_tracker.start()
            seed_python_numpy(seed)
            predict_start = time.time()
            try:
                annotated = predict_fn(
                    test_raw, model_handle, pred_key, seed,
                    run_out_path, test_args,
                )
            except Exception:
                predict_tracker.stop()
                raise
            predict_runtime = time.time() - predict_start
            predict_memory = predict_tracker.stop()

            if list(annotated.obs_names) != list(test_raw.obs_names):
                if set(annotated.obs_names) != set(test_raw.obs_names):
                    raise ValueError(
                        "Prediction output cells do not match the test dataset"
                    )
                annotated = annotated[test_raw.obs_names].copy()
            if pred_key not in annotated.obs:
                raise KeyError(
                    f"Method did not create prediction column '{pred_key}'"
                )
            method_obs = annotated.obs.copy()
            annotated.obs = test_metadata.reindex(annotated.obs_names).copy()
            for column in method_obs:
                if column not in annotated.obs or column == pred_key:
                    annotated.obs[column] = method_obs[column].reindex(
                        annotated.obs_names
                    )
            annotated.obs[pred_key] = (
                annotated.obs[pred_key]
                .astype("string").fillna("Unknown").astype(str)
            )

            if test_args.test_celltype is not None:
                metrics, per_class, confusion = evaluate_predictions(
                    annotated,
                    pred_key,
                    test_args.test_celltype,
                    dataset_name=spec["name"],
                )
                evaluation_status = "computed"
            else:
                metrics, per_class, confusion = {}, None, None
                evaluation_status = "skipped_no_test_celltype"

            metrics.update({
                "tool": tool_name,
                "run_id": run_id,
                "seed": seed,
                "evaluation_status": evaluation_status,
                "train_runtime_seconds": train_runtime,
                "predict_runtime_seconds": predict_runtime,
                "runtime_seconds": train_runtime + predict_runtime,
                "peak_ram_mb": max(
                    train_memory["peak_ram_mb"],
                    predict_memory["peak_ram_mb"],
                ),
                "peak_vram_mb": max(
                    train_memory["peak_vram_mb"],
                    predict_memory["peak_vram_mb"],
                ),
            })
            _write_run_results(
                annotated, metrics, per_class, confusion, tool_name, pred_key,
                test_args, seed, run_out_path,
            )
            _write_json_atomic(
                run_out_path / _RUN_COMPLETE_FILE,
                _run_marker_payload(
                    "complete", tool_name, run_id, seed,
                    test_args, spec["path"],
                ),
            )
            progress_marker_path.unlink()

            if evaluation_status == "computed":
                print(
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"F1(macro)={metrics['f1_macro']:.4f}, "
                    f"AUC(macro)={metrics['auc_ovr_macro']:.4f}, "
                    f"Exact-match audit={metrics['exact_match_accuracy']:.4f}"
                )
            print(f"Results saved to {run_out_path}")
            del (
                annotated,
                confusion,
                method_obs,
                metrics,
                per_class,
                test_metadata,
                test_raw,
            )
            gc.collect()

        del model_handle
        gc.collect()

        all_complete = all(
            _run_is_complete(
                tool_name,
                run_id,
                seed,
                spec["save_path"] / tool_name / str(run_id),
                spec["args"],
                spec["path"],
            )
            for spec in test_specs
        )
        if all_complete and not args.save_models:
            _cleanup_model_artifacts(shared_run_path, multi_test)

    del train_raw
    gc.collect()
    _refresh_metric_summaries(test_specs, tool_name, args.run_times)
    print(
        f"\nAll {tool_name} runs completed for "
        f"{len(test_specs)} test dataset(s)."
    )
