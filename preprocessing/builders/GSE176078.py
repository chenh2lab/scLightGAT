from __future__ import annotations

import pandas as pd

from builders.common import (
    add_data_source,
    download_geo_dataset,
    find,
    subset_to_metadata,
)
from utils.config import GEO_PATTERNS
from utils.donor import attach_donor_metadata
from utils.io import read_10x_mtx


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE176078",
        "GSE176078",
        rawdir,
        GEO_PATTERNS,
    )
    root = source / "extracted"

    # The Wu et al. GEO archive contains one atlas-level sparse matrix with
    # matching barcodes/genes and one atlas-level metadata.csv.  The first CSV
    # column is the published cell ID (commonly named `X` when read in Python).
    adata = read_10x_mtx(
        find(root, r"^count_matrix_sparse\.mtx$"),
        find(root, r"^count_matrix_barcodes\.tsv$"),
        find(root, r"^count_matrix_genes\.tsv$"),
    )
    metadata_path = find(root, r"^metadata\.csv$")
    metadata = pd.read_csv(metadata_path, index_col=0, low_memory=False)
    metadata.index = metadata.index.astype(str)

    adata, matched, source_column = subset_to_metadata(adata, metadata)
    print(
        f"  GSE176078 metadata matching: {matched:,}/{len(metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )
    adata = add_data_source(adata, "GSE176078")
    return attach_donor_metadata(adata, "GSE176078")
