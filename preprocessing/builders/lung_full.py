from __future__ import annotations

import pandas as pd

from builders.common import (
    add_data_source,
    download_geo_dataset,
    find,
    subset_to_metadata,
)
from utils.config import GEO_PATTERNS
from utils.io import read_10x_mtx


def build(rawdir, auxdir):
    """Rebuild the scDiffusion lung benchmark from GSE135893 raw counts."""
    source = download_geo_dataset(
        "lung_full",
        "GSE135893",
        rawdir,
        GEO_PATTERNS,
    )
    adata = read_10x_mtx(
        find(source, r"GSE135893_matrix\.mtx\.gz$"),
        find(source, r"GSE135893_barcodes\.tsv\.gz$"),
        find(source, r"GSE135893_genes\.tsv\.gz$"),
    )
    metadata = pd.read_csv(
        find(source, r"GSE135893_IPF_metadata\.csv\.gz$"),
        index_col=0,
        low_memory=False,
    )
    metadata.index = metadata.index.astype(str)
    adata, matched, source_column = subset_to_metadata(adata, metadata)
    if matched != adata.n_obs:
        raise RuntimeError(
            f"lung_full metadata matched {matched:,}/{adata.n_obs:,} expression cells"
        )
    if "celltype" not in adata.obs:
        raise RuntimeError("GSE135893 metadata has no celltype column")
    adata.obs["source_celltype"] = adata.obs["celltype"].astype(str).str.strip()
    adata.uns["upstream_provenance"] = {
        "study": "Habermann human lung pulmonary-fibrosis dataset",
        "accession": "GSE135893",
        "expression_source": "GEO raw MatrixMarket counts",
        "metadata_id_source": str(source_column),
    }
    return add_data_source(adata, "lung_full")
