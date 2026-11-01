from __future__ import annotations

import pandas as pd

from builders.common import (
    add_data_source,
    attach_metadata_by_ids,
    download_geo_dataset,
    find,
)
from utils.config import GEO_PATTERNS
from utils.io import read_wide_counts


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE115978",
        "GSE115978",
        rawdir,
        GEO_PATTERNS,
    )
    adata = read_wide_counts(find(source, r"counts\.csv"), sep=",")

    # GEO documents one metadata row per cell; the first column is the cell
    # name and matches the expression-matrix columns.
    metadata = pd.read_csv(
        find(source, r"cell\.annotations"),
        index_col=0,
        low_memory=False,
    )
    metadata.index = metadata.index.astype(str)

    adata, matched, source_column = attach_metadata_by_ids(adata, metadata)
    if matched == 0:
        raise RuntimeError("GSE115978 GEO annotations did not match expression cells")
    print(
        f"  GSE115978 metadata matching: {matched:,}/{len(metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )

    # Expression and annotation both come from the original GEO study.
    return add_data_source(adata, "GSE115978")
