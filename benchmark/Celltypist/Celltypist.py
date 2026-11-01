"""CellTypist reference-training cell type annotation benchmark."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import celltypist
import numpy as np
import pandas as pd
import scanpy as sc

from utils.dataset import is_count_matrix
from utils.pipeline import (
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
    seed_python_numpy,
)


def _normalize_log1p(adata, target_sum=1e4, *, copy=True):
    """Apply CellTypist input normalization without mutating shared data."""
    out = adata.copy() if copy else adata
    if is_count_matrix(out.X):
        sc.pp.normalize_total(out, target_sum=target_sum)
        sc.pp.log1p(out)
    return out


def train_celltypist(adata, seed, run_out_path, args):
    """Train CellTypist on log-normalized reference expression."""
    seed_python_numpy(seed)
    # Training receives a lightweight AnnData whose expression matrix is
    # shared with the immutable reference. Copying here keeps normalization
    # inside the common per-run timing boundary without mutating later runs.
    reference = _normalize_log1p(adata, copy=True)
    model = celltypist.train(
        X=reference,
        labels=args.train_celltype,
        n_jobs=args.n_jobs,
        random_state=seed,
    )
    if args.save_models:
        model_dir = run_out_path / "_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.write(str(model_dir / "model.pkl"))
    return model


PREDICTION_MEMORY_LIMIT_BYTES = 23 * 1024**3
PREDICTION_MEMORY_RESERVE_BYTES = 2 * 1024**3
PREDICTION_FALLBACK_CHUNK_SIZE = 2048


def _process_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _prediction_chunk_size(query, model):
    model_features = np.asarray(model.classifier.features, dtype=str)
    query_features = query.var_names.astype(str).to_numpy()
    overlap = int(np.isin(query_features, model_features).sum())
    if overlap == 0:
        return query.n_obs

    # CellTypist centers and scales the matched sparse expression into dense
    # float64 working arrays. Allow three such arrays plus a fixed reserve.
    bytes_per_cell = 3 * overlap * np.dtype(np.float64).itemsize
    resident = _process_rss_bytes()
    fixed = resident + PREDICTION_MEMORY_RESERVE_BYTES
    full_estimate = fixed + query.n_obs * bytes_per_cell
    if full_estimate <= PREDICTION_MEMORY_LIMIT_BYTES:
        return query.n_obs
    if fixed >= PREDICTION_MEMORY_LIMIT_BYTES:
        return min(query.n_obs, PREDICTION_FALLBACK_CHUNK_SIZE)
    available = PREDICTION_MEMORY_LIMIT_BYTES - fixed
    return max(1, min(query.n_obs, available // bytes_per_cell))


def _annotate_query(query, model):
    """Run the official predictor, batching only for the memory limit."""
    if query.n_obs == 0:
        raise ValueError("Query dataset contains no cells")

    chunk_size = int(_prediction_chunk_size(query, model))
    if chunk_size >= query.n_obs:
        result = celltypist.annotate(filename=query, model=model)
        return result.predicted_labels.copy(), result.probability_matrix.copy()

    print(
        f"[Celltypist] memory fallback: predicting {query.n_obs} cells "
        f"in chunks of {chunk_size}"
    )

    predicted_parts = []
    probability_parts = []
    for start in range(0, query.n_obs, chunk_size):
        chunk = query[start : start + chunk_size].copy()
        result = celltypist.annotate(filename=chunk, model=model)
        predicted_parts.append(result.predicted_labels.copy())
        probability_parts.append(result.probability_matrix.copy())

    return pd.concat(predicted_parts), pd.concat(probability_parts)


def predict_celltypist(adata, model, pred_key, seed, run_out_path, args):
    """Annotate query cells and retain CellTypist class probabilities."""
    query = _normalize_log1p(adata, copy=False)
    predicted, probabilities = _annotate_query(query, model)

    predicted.index = predicted.index.astype(str)
    adata.obs[pred_key] = (
        predicted["predicted_labels"]
        .reindex(adata.obs_names.astype(str))
        .astype("string")
        .fillna("Unknown")
        .astype(str)
        .values
    )

    probabilities.index = probabilities.index.astype(str)
    probabilities = probabilities.reindex(adata.obs_names.astype(str))
    adata.obsm[SCORE_KEY] = probabilities.to_numpy(dtype=float)
    adata.uns[SCORE_CLASSES_KEY] = np.asarray(probabilities.columns, dtype=str)
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("Celltypist", include_n_jobs=True)
    arguments = parser.parse_args()
    run_annotation_pipeline(
        "Celltypist",
        "celltypist_prediction",
        train_celltypist,
        predict_celltypist,
        arguments,
        expression_layers=(),
        share_train_expression=True,
        keep_var_metadata=False,
    )
