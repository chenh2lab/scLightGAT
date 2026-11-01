"""Expression graph construction and GAT tensor preparation."""

import logging

import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from .hparam import (
    GRAPH_MAX_PCA_COMPONENTS,
    GRAPH_METRIC,
    GRAPH_N_NEIGHBORS,
    HARMONY_MAX_ITERATIONS,
)


logger = logging.getLogger(__name__)


def build_neighbor_graph(
    adata,
    batch_key=None,
    seed=0,
    n_neighbors=None,
    metric=None,
):
    """Build the PCA/Harmony neighbor graph consumed by GAT."""
    n_neighbors = int(
        GRAPH_N_NEIGHBORS if n_neighbors is None else n_neighbors
    )
    metric = GRAPH_METRIC if metric is None else str(metric)
    n_components = min(
        int(GRAPH_MAX_PCA_COMPONENTS),
        adata.n_obs - 1,
        adata.n_vars - 1,
    )
    if n_components < 1:
        raise ValueError(
            f"PCA requires at least 2 cells and 2 genes; got {adata.shape}"
        )
    sc.pp.pca(
        adata,
        n_comps=n_components,
        mask_var=None,
        random_state=seed,
    )

    if (
        batch_key
        and batch_key in adata.obs
        and adata.obs[batch_key].nunique() > 1
    ):
        harmony = hm.run_harmony(
            adata.obsm["X_pca"],
            adata.obs,
            [batch_key],
            max_iter_harmony=int(HARMONY_MAX_ITERATIONS),
            plot_convergence=True,
            random_state=seed,
        )
        adata.obsm["X_pca_harmony"] = pd.DataFrame(harmony.Z_corr).T.values
        representation = "X_pca_harmony"
    else:
        representation = "X_pca"

    sc.pp.neighbors(
        adata,
        use_rep=representation,
        n_neighbors=min(n_neighbors, max(1, int(adata.n_obs) - 1)),
        metric=metric,
        random_state=seed,
    )
    logger.info(
        "Graph construction completed (neighbors=%d, metric=%s)",
        n_neighbors,
        metric,
    )
    return adata


def build_gat_data(adata, lightgbm_probabilities):
    """Combine expression and OOF/predicted probabilities as node features."""
    if "log_transformed" not in adata.layers:
        adata.layers["log_transformed"] = np.log1p(adata.X)
    expression = adata.to_df(layer="log_transformed").values
    node_features = np.hstack(
        [expression, np.asarray(lightgbm_probabilities)]
    )
    connectivities = adata.obsp["connectivities"].tocoo()
    edge_index = torch.tensor(
        np.vstack([connectivities.row, connectivities.col]),
        dtype=torch.long,
    )
    edge_weights = torch.tensor(connectivities.data, dtype=torch.float)
    edge_index, _ = to_undirected(
        edge_index,
        edge_weights,
        num_nodes=adata.n_obs,
        reduce="max",
    )
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
    )
