"""Dataset utilities shared by cell type annotation benchmarks."""

from __future__ import annotations

from anndata import AnnData
from anndata.io import read_elem
import h5py
import numpy as np
import scanpy as sc
from scipy import sparse


def load_dataset(
    path, *, obs_columns=None, layers=None, keep_var_metadata=True,
):
    """Load only the H5AD elements required by an annotation method.

    ``X`` and every cell/gene are always loaded without sampling.  Optional
    ``obs_columns`` and ``layers`` limit auxiliary data at the storage read,
    avoiding materializing ``raw``, embeddings, graphs, and unused expression
    layers only to discard them later. ``keep_var_metadata=False`` retains the
    exact gene index while omitting method-irrelevant ``var`` columns.
    """
    print(f"Loading data from {path}")
    requested_obs = None
    if obs_columns is not None:
        requested_obs = list(dict.fromkeys(
            str(column) for column in obs_columns if column is not None
        ))
    requested_layers = None
    if layers is not None:
        requested_layers = list(dict.fromkeys(str(layer) for layer in layers))

    with h5py.File(path, "r") as store:
        if "X" not in store:
            raise ValueError(f"Dataset has no expression matrix X: {path}")
        obs = read_elem(store["obs"])
        if requested_obs is not None:
            missing = [column for column in requested_obs if column not in obs]
            if missing:
                raise KeyError(
                    "Requested observation columns are missing from "
                    f"{path}: {', '.join(missing)}"
                )
            obs = obs.loc[:, requested_obs].copy()

        var = read_elem(store["var"])
        if not keep_var_metadata:
            var = var.iloc[:, 0:0].copy()

        layer_group = store.get("layers")
        available_layers = [] if layer_group is None else list(layer_group.keys())
        selected_layers = (
            available_layers
            if requested_layers is None
            else [layer for layer in requested_layers if layer in available_layers]
        )
        adata = AnnData(
            X=read_elem(store["X"]),
            obs=obs,
            var=var,
            layers={
                layer: read_elem(layer_group[layer])
                for layer in selected_layers
            },
        )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def model_input_with_shared_expression(adata, obs_columns):
    """Create isolated metadata while sharing read-only expression matrices.

    scLightGAT treats ``X`` and input layers as immutable.  Sharing those
    matrices prevents a full deep copy for every seed while fresh ``obs`` and
    ``var`` frames keep method-created annotations isolated between runs.
    """
    columns = list(dict.fromkeys(
        str(column) for column in obs_columns if column is not None
    ))
    return AnnData(
        X=adata.X,
        obs=adata.obs.loc[:, columns].copy(),
        var=adata.var.copy(),
        layers={name: matrix for name, matrix in adata.layers.items()},
    )


def is_count_matrix(matrix, sample_size=10000, tolerance=1e-6):
    """Return True when sampled non-zero values look like raw integer counts."""
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        return False
    if values.size > sample_size:
        indices = np.linspace(0, values.size - 1, sample_size, dtype=int)
        values = values[indices]
    values = np.asarray(values, dtype=float)
    return bool(
        np.all(np.isfinite(values))
        and np.min(values) >= 0
        and np.all(np.abs(values - np.rint(values)) <= tolerance)
    )


def normalize_log1p(adata, target_sum=1e4, *, copy=True):
    """Normalize raw counts to 10,000 counts/cell and log1p-transform.

    Matrices that already contain non-integer values are treated as processed
    expression and returned unchanged. Set copy=False only when the caller
    already owns a disposable AnnData copy.
    """
    out = adata.copy() if copy else adata
    if is_count_matrix(out.X):
        sc.pp.normalize_total(out, target_sum=target_sum)
        sc.pp.log1p(out)
    return out


def prepare_umap(adata, seed, n_top_genes=2000):
    """Compute a test-expression UMAP without dropping or reordering cells."""
    # This runs after prediction/evaluation on a disposable per-run AnnData.
    # Reusing it avoids another full copy of a potentially very large query.
    vis = normalize_log1p(adata, copy=False)

    if vis.n_obs < 3:
        raise ValueError("At least three test cells are required to compute UMAP")

    nonzero = np.asarray((vis.X != 0).sum(axis=0)).ravel() > 0
    if not np.all(nonzero):
        vis = vis[:, nonzero].copy()
    if vis.n_vars < 2:
        raise ValueError("At least two expressed genes are required to compute UMAP")

    if vis.n_vars > n_top_genes:
        try:
            sc.pp.highly_variable_genes(
                vis, n_top_genes=n_top_genes, flavor="seurat", subset=True
            )
        except Exception as exc:
            print(f"[UMAP] HVG selection failed; using all genes: {exc}")

    n_comps = min(50, vis.n_obs - 1, vis.n_vars - 1)
    sc.tl.pca(vis, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(
        vis,
        use_rep="X_pca",
        n_neighbors=min(15, vis.n_obs - 1),
        random_state=seed,
    )
    sc.tl.umap(vis, random_state=seed)
    return vis
