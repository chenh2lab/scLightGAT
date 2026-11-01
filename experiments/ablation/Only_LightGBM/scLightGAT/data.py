"""Reference-only feature fitting and train/query schema alignment."""

import itertools

from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from .hparam import (
    DGE_ADJUSTED_PVALUE,
    DGE_FEATURES_PER_TYPE,
    DGE_LOG_FOLD_CHANGE,
    DGE_MAX_FEATURES_PER_TYPE,
    DGE_METHOD,
    DGE_MIN_FEATURES_PER_TYPE,
)


def get_celltype_column(adata, celltype_column=None):
    if celltype_column is not None and celltype_column in adata.obs:
        return celltype_column
    for candidate in ("Ground Truth", "Celltype_training"):
        if candidate in adata.obs:
            return candidate
    available = ", ".join(adata.obs.columns)
    raise ValueError(
        f"No valid cell-type column found. Available columns: {available}"
    )


def _rank_markers(adata, group):
    sc.tl.rank_genes_groups(
        adata,
        groupby=group,
        method=DGE_METHOD,
        reference="rest",
        use_raw=False,
        layer="log_transformed",
    )
    markers = sc.get.rank_genes_groups_df(adata, None)
    markers = markers[markers["logfoldchanges"] > 0].copy()
    markers["passes_threshold"] = (
        (markers["pvals_adj"] < DGE_ADJUSTED_PVALUE)
        & (markers["logfoldchanges"] > DGE_LOG_FOLD_CHANGE)
    )
    markers["z_score_rank"] = markers.groupby(
        "group", observed=False
    )["scores"].rank(ascending=False)
    markers["logFC_rank"] = markers.groupby(
        "group", observed=False
    )["logfoldchanges"].rank(ascending=False)
    markers["combined_rank"] = (
        markers["z_score_rank"] + markers["logFC_rank"]
    ) / 2
    return markers.sort_values(
        ["group", "combined_rank"], ascending=[True, True]
    ).reset_index(drop=True)


def _features_per_celltype(markers, requested_top):
    maximum = min(
        int(requested_top), int(DGE_MAX_FEATURES_PER_TYPE)
    )
    minimum = int(DGE_MIN_FEATURES_PER_TYPE)
    selected = {group: [] for group in markers["group"].unique()}

    for group in selected:
        group_markers = markers[markers["group"] == group]
        genes = (
            group_markers[group_markers["passes_threshold"]]["names"]
            .drop_duplicates()
            .head(maximum)
            .tolist()
        )
        if len(genes) < minimum:
            remaining = group_markers[
                ~group_markers["names"].isin(genes)
            ]["names"].drop_duplicates().tolist()
            genes.extend(remaining[:max(0, minimum - len(genes))])
        selected[group] = genes[:maximum]
    return selected


def select_dge_genes(adata, celltype_column=None, top_features=None):
    """Fit the ordered DGE feature schema on labelled reference cells."""
    if top_features is None:
        top_features = DGE_FEATURES_PER_TYPE
    group = get_celltype_column(adata, celltype_column)
    per_type = _features_per_celltype(
        _rank_markers(adata, group), top_features
    )
    ordered = list(
        dict.fromkeys(itertools.chain.from_iterable(per_type.values()))
    )
    available = set(adata.var_names.astype(str))
    return [gene for gene in ordered if gene in available]


def _matrix_column_means(matrix):
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).reshape(-1)
    else:
        means = np.asarray(matrix).mean(axis=0).reshape(-1)
    means = means.astype(np.float32, copy=False)
    if not np.isfinite(means).all():
        raise ValueError("Reference matrix contains non-finite gene means")
    return means


def compute_reference_gene_means(adata, genes):
    """Calculate train-only imputation values in each matrix's own scale."""
    genes = [str(gene) for gene in genes]
    if not genes:
        raise ValueError("The fitted feature schema is empty")
    if len(set(genes)) != len(genes):
        raise ValueError("The fitted feature schema contains duplicate genes")

    source_names = adata.var_names.astype(str).tolist()
    source_lookup = {gene: index for index, gene in enumerate(source_names)}
    missing = [gene for gene in genes if gene not in source_lookup]
    if missing:
        raise ValueError(
            "Reference data are missing genes from their own schema: "
            + ", ".join(missing[:10])
        )
    indices = [source_lookup[gene] for gene in genes]
    means = {"X": _matrix_column_means(adata.X[:, indices])}
    for layer in ("counts", "log_transformed"):
        if layer in adata.layers:
            means[layer] = _matrix_column_means(
                adata.layers[layer][:, indices]
            )
    return means


def _missing_gene_mask(source_columns, target_columns):
    source_set = {str(gene) for gene in source_columns}
    return np.fromiter(
        (str(gene) not in source_set for gene in target_columns),
        dtype=bool,
        count=len(target_columns),
    )


def _align_matrix(
    matrix,
    source_columns,
    target_columns,
    fill_values=None,
    matrix_name="matrix",
):
    source_lookup = {
        str(gene): index for index, gene in enumerate(source_columns)
    }
    present_target = []
    present_source = []
    missing_target = []
    for target_index, gene in enumerate(target_columns):
        source_index = source_lookup.get(str(gene))
        if source_index is None:
            missing_target.append(target_index)
        else:
            present_target.append(target_index)
            present_source.append(source_index)

    selected = sparse.csr_matrix(matrix[:, present_source])
    if not missing_target:
        return selected
    if fill_values is None:
        missing_genes = [target_columns[index] for index in missing_target]
        raise ValueError(
            f"{matrix_name} is missing {len(missing_genes)} fitted genes but "
            "no train reference means were supplied: "
            + ", ".join(missing_genes[:10])
        )

    fill_values = np.asarray(fill_values, dtype=np.float32).reshape(-1)
    if fill_values.shape != (len(target_columns),):
        raise ValueError(
            f"{matrix_name} reference means have shape {fill_values.shape}; "
            f"expected ({len(target_columns)},)"
        )
    if not np.isfinite(fill_values).all():
        raise ValueError(
            f"{matrix_name} reference means contain non-finite values"
        )

    coo = selected.tocoo()
    present_target = np.asarray(present_target, dtype=np.int64)
    mapped_columns = (
        present_target[coo.col]
        if coo.nnz
        else np.empty(0, dtype=np.int64)
    )
    result_dtype = np.result_type(selected.dtype, fill_values.dtype)
    row_parts = [coo.row.astype(np.int64, copy=False)]
    column_parts = [mapped_columns]
    data_parts = [coo.data.astype(result_dtype, copy=False)]

    missing_target = np.asarray(missing_target, dtype=np.int64)
    missing_values = fill_values[missing_target].astype(
        result_dtype, copy=False
    )
    nonzero = missing_values != 0
    if nonzero.any() and matrix.shape[0] > 0:
        fill_columns = missing_target[nonzero]
        fill_constants = missing_values[nonzero]
        row_parts.append(
            np.repeat(
                np.arange(matrix.shape[0], dtype=np.int64),
                len(fill_columns),
            )
        )
        column_parts.append(np.tile(fill_columns, matrix.shape[0]))
        data_parts.append(np.tile(fill_constants, matrix.shape[0]))

    return sparse.csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(matrix.shape[0], len(target_columns)),
        dtype=result_dtype,
    )


def align_adata_to_genes(adata, genes, reference_means=None):
    """Project a query to a fitted schema with train-mean imputation."""
    genes = [str(gene) for gene in genes]
    if not genes:
        raise ValueError("The fitted feature schema is empty")
    if len(set(genes)) != len(genes):
        raise ValueError("The fitted feature schema contains duplicate genes")
    if reference_means is not None and not hasattr(reference_means, "get"):
        raise TypeError("reference_means must be keyed by matrix name")

    means = {} if reference_means is None else reference_means
    source_names = adata.var_names.astype(str).tolist()
    missing_mask = _missing_gene_mask(source_names, genes)
    var = adata.var.copy()
    var.index = pd.Index(source_names)
    var = var.reindex(genes)
    var["scLightGAT_missing_in_source"] = missing_mask
    aligned = AnnData(
        X=_align_matrix(
            adata.X,
            source_names,
            genes,
            fill_values=means.get("X"),
            matrix_name="X",
        ),
        obs=adata.obs.copy(),
        var=var,
    )
    for layer in ("counts", "log_transformed"):
        if layer in adata.layers:
            aligned.layers[layer] = _align_matrix(
                adata.layers[layer],
                source_names,
                genes,
                fill_values=means.get(layer),
                matrix_name=f"layer '{layer}'",
            )
    aligned.uns["scLightGAT_alignment_imputation"] = (
        "train_reference_gene_mean_per_matrix_scale"
    )
    return aligned


def _expression_frame(adata, celltype_column):
    if "log_transformed" not in adata.layers:
        adata.layers["log_transformed"] = np.log1p(adata.X)
    frame = adata.to_df(layer="log_transformed")
    frame["Cell_type"] = adata.obs[celltype_column]
    return frame


def prepare_lightgbm_data(adata, celltype_column=None):
    celltype_column = get_celltype_column(adata, celltype_column)
    frame = _expression_frame(adata, celltype_column)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    encoder = LabelEncoder()
    frame["encoding"] = encoder.fit_transform(frame["Cell_type"])
    features = frame.drop(["Cell_type", "encoding"], axis=1)
    features.columns = features.columns.astype(str)
    return features, frame["encoding"], encoder

