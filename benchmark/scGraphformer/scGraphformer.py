#!/usr/bin/env python3
"""Official scGraphformer class adapted to the shared annotation pipeline."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import types
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, remove_self_loops, subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.dataset import is_count_matrix
from utils.pipeline import (  # noqa: E402
    SCORE_CLASSES_KEY,
    SCORE_KEY,
    create_argument_parser,
    run_annotation_pipeline,
    seed_python_numpy,
)


HVG_COUNT = 3500
KNN_NEIGHBORS = 8

# Fixed to the authors' cross-platform workflow.  These values are still
# attached to the shared argparse namespace so the pipeline's training
# signature remains identical to runs made with the former CLI defaults.
SCGRAPHFORMER_SETTINGS = {
    "epochs": 20,
    "batch_size": 256,
    "train_prop": 0.8,
    "num_layers": 1,
    "num_heads": 2,
    "alpha": 0.5,
    "dropout": 0.3,
    "lr": 0.0005,
    "weight_decay": 0.05,
    "device": 0,
    "cpu": False,
    "use_graph": True,
    "use_knn": True,
    "use_HVG": True,
}


class _SparseTensor:
    """Minimal holder for the official model's torch_sparse call."""

    def __init__(self, row, col, value=None, sparse_sizes=None):
        self.row, self.col = row, col
        self.value, self.sparse_sizes = value, sparse_sizes


def _sparse_matmul(adjacency, features):
    output = features.new_zeros(
        (adjacency.sparse_sizes[0], *features.shape[1:])
    )
    messages = features.index_select(0, adjacency.col)
    if adjacency.value is not None:
        shape = (-1,) + (1,) * (features.ndim - 1)
        messages *= adjacency.value.reshape(shape).to(features.dtype)
    return output.index_add_(0, adjacency.row, messages)


def _install_sparse_fallback():
    try:
        __import__("torch_sparse")
        return
    except (ImportError, OSError):
        if platform.machine().lower() not in {"aarch64", "arm64"}:
            raise
    sys.modules.pop("torch_sparse", None)
    module = types.ModuleType("torch_sparse")
    module.SparseTensor, module.matmul = _SparseTensor, _sparse_matmul
    sys.modules["torch_sparse"] = module


def _official_model_class():
    default = Path(sys.prefix) / "src" / "scGraphformer"
    home = Path(os.environ.get("SCGRAPHFORMER_HOME", default)).expanduser()
    source = home / "utils" / "scGraphformer.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Official source not found at {source}; clone the pinned "
            "scGraphformer revision or set SCGRAPHFORMER_HOME"
        )
    _install_sparse_fallback()
    spec = importlib.util.spec_from_file_location(
        "_official_scgraphformer", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.scGraphformer


def _seed(seed, n_jobs):
    seed_python_numpy(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(n_jobs)))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _dense(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.nan_to_num(np.asarray(matrix, dtype=np.float32), copy=False)


def _normalize_log1p(adata, target_sum=1e4, *, copy=True):
    """Apply the official input normalization without mutating shared data."""
    out = adata.copy() if copy else adata
    if is_count_matrix(out.X):
        sc.pp.normalize_total(out, target_sum=target_sum)
        sc.pp.log1p(out)
    return out


def _expressed_gene_mask(matrix):
    """Return genes with at least one non-zero reference value."""
    if sparse.issparse(matrix):
        return np.asarray((matrix != 0).sum(axis=0)).ravel() > 0
    return np.any(np.asarray(matrix) != 0, axis=0)


def _select_reference_hvgs(reference):
    """Run the official HVG selection with its zero-gene edge case handled."""
    kwargs = {
        "min_mean": 0.0125,
        "max_mean": 4,
        "min_disp": 0.5,
        "n_top_genes": HVG_COUNT,
        "flavor": "cell_ranger",
        "subset": True,
    }
    try:
        sc.pp.highly_variable_genes(reference, **kwargs)
        return reference
    except ValueError as error:
        if "Bin edges must be unique" not in str(error):
            raise

    expressed = _expressed_gene_mask(reference.X)
    removed = int((~expressed).sum())
    if removed == 0:
        raise RuntimeError(
            "scGraphformer HVG binning failed although the reference has no "
            "all-zero genes"
        )
    if not expressed.any():
        raise ValueError("Reference dataset contains no expressed genes")
    print(
        f"[scGraphformer] HVG binning encountered tied zero-expression "
        f"genes; removing {removed} reference-only all-zero genes and retrying"
    )
    reference = reference[:, expressed].copy()
    if reference.n_vars > HVG_COUNT:
        sc.pp.highly_variable_genes(reference, **kwargs)
    return reference


def _reference_features(adata, use_hvg):
    reference = _normalize_log1p(adata, copy=True)
    if reference.n_vars == 0:
        raise ValueError("Reference dataset contains no genes")
    if use_hvg and reference.n_vars > HVG_COUNT:
        reference = _select_reference_hvgs(reference)
    genes = reference.var_names.astype(str).to_numpy()
    return torch.from_numpy(_dense(reference.X)), genes


def _query_features(adata, genes):
    query = _normalize_log1p(adata, copy=True)
    locations = query.var_names.astype(str).get_indexer(genes)
    present = locations >= 0
    if not present.any():
        raise ValueError("Query has no genes in common with the reference")
    aligned = np.zeros((query.n_obs, len(genes)), dtype=np.float32)
    aligned[:, present] = _dense(query.X[:, locations[present]])
    return torch.from_numpy(aligned)


def _knn_edges(features, n_jobs):
    adjacency = kneighbors_graph(
        features.numpy(),
        n_neighbors=min(KNN_NEIGHBORS, features.shape[0]),
        include_self=True,
        n_jobs=max(1, int(n_jobs)),
    )
    edges = torch.as_tensor(np.vstack(adjacency.nonzero()), dtype=torch.long)
    edges, _ = remove_self_loops(edges)
    edges, _ = add_self_loops(edges, num_nodes=features.shape[0])
    return edges


def train_scgraphformer(adata, seed, run_out_path, args):
    """Fit one official model on the labelled reference."""
    del run_out_path
    if not 0 < args.train_prop <= 1:
        raise ValueError("Fixed scGraphformer train_prop must be in (0, 1]")
    if args.batch_size < 2:
        raise ValueError("Fixed scGraphformer batch_size must be at least 2")
    if args.use_graph and not args.use_knn:
        raise ValueError("Fixed scGraphformer use_graph requires use_knn")
    _seed(seed, args.n_jobs)
    device_name = "cpu" if args.cpu or not torch.cuda.is_available() else (
        f"cuda:{args.device}"
    )
    device = torch.device(device_name)
    features, genes = _reference_features(adata, args.use_HVG)
    encoder = LabelEncoder()
    raw_labels = adata.obs[args.train_celltype].astype(str).to_numpy()
    labels = torch.as_tensor(encoder.fit_transform(raw_labels), dtype=torch.long)
    train_size = (
        adata.n_obs
        if args.train_prop == 1
        else int(adata.n_obs * args.train_prop)
    )
    if train_size < 2:
        raise ValueError("scGraphformer needs at least two training cells")
    chosen = np.random.choice(adata.n_obs, train_size, replace=False)
    train_indices = torch.as_tensor(chosen)
    edges = _knn_edges(features, args.n_jobs) if args.use_knn else None
    model = _official_model_class()(
        features.shape[1],
        len(encoder.classes_),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        alpha=args.alpha,
        dropout=args.dropout,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_graph=args.use_graph,
    ).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        shuffled = train_indices[torch.randperm(train_indices.numel())]
        batches = list(shuffled.split(args.batch_size))
        if len(batches) > 1 and batches[-1].numel() == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        losses = []
        for batch in batches:
            batch_edges = None
            if edges is not None:
                batch_edges, _ = subgraph(
                    batch,
                    edges,
                    relabel_nodes=True,
                    num_nodes=features.shape[0],
                )
                batch_edges = batch_edges.to(device)
            optimizer.zero_grad()
            logits = model(features[batch].to(device), batch_edges)
            output = F.log_softmax(logits, dim=1)
            loss = criterion(output, labels[batch].to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        print(
            f"[scGraphformer] epoch {epoch + 1:02d}/{args.epochs} "
            f"loss={np.mean(losses):.6f}"
        )
    return {
        "model": model,
        "genes": genes,
        "classes": encoder.classes_.astype(str),
        "device": device,
    }


def predict_scgraphformer(
    adata, model_handle, pred_key, seed, run_out_path, args
):
    del seed, run_out_path
    if adata.n_obs == 0:
        raise ValueError("Query dataset contains no cells")
    features = _query_features(adata, model_handle["genes"])
    edges = _knn_edges(features, args.n_jobs) if args.use_knn else None
    model, device = model_handle["model"], model_handle["device"]
    model.eval()
    with torch.no_grad():
        device_edges = None if edges is None else edges.to(device)
        logits = model(features.to(device), device_edges)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    classes = model_handle["classes"]
    adata.obs[pred_key] = classes[probabilities.argmax(axis=1)]
    adata.obsm[SCORE_KEY] = probabilities
    adata.uns[SCORE_CLASSES_KEY] = classes
    adata.uns["annotation_score_type"] = "probabilities"
    return adata


if __name__ == "__main__":
    parser = create_argument_parser("scGraphformer", include_n_jobs=True)
    parser.set_defaults(**SCGRAPHFORMER_SETTINGS)
    args = parser.parse_args()
    run_annotation_pipeline(
        "scGraphformer",
        "scgraphformer_prediction",
        train_scgraphformer,
        predict_scgraphformer,
        args,
        expression_layers=(),
        share_train_expression=True,
        keep_var_metadata=False,
    )
