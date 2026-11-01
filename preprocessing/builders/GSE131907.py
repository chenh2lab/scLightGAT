from __future__ import annotations

from builders.common import (
    add_data_source,
    download_geo_dataset,
    find,
    subset_to_metadata,
)
from utils.config import GEO_PATTERNS
from utils.donor import attach_donor_metadata
from utils.io import read_table, read_wide_counts


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE131907",
        "GSE131907",
        rawdir,
        GEO_PATTERNS,
    )
    adata = read_wide_counts(find(source, r"raw_UMI_matrix\.txt\.gz$"))
    metadata = read_table(find(source, r"cell_annotation\.txt\.gz$"))

    # The published annotation table uses `Sample` for sample identity and
    # `Index` as the unique cell ID matching raw UMI matrix columns.
    required = {"Sample", "Index"}
    missing = required - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"GSE131907 metadata is missing required columns {sorted(missing)}; "
            f"available={list(metadata.columns)}"
        )

    # Thesis reconstruction: retain primary tumor (T) and distant-normal (N)
    # lung samples only.
    keep = metadata["Sample"].astype(str).str.fullmatch(
        r"LUNG_[NT]\d+",
        na=False,
    )
    metadata = metadata.loc[keep].copy()
    if metadata.empty:
        raise RuntimeError("GSE131907: lung tumor/normal sample selection is empty")

    adata, matched, source_column = subset_to_metadata(
        adata,
        metadata,
        ["Index"],
    )
    print(
        f"  GSE131907 metadata matching: {matched:,}/{len(metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )
    adata = add_data_source(adata, "GSE131907")
    return attach_donor_metadata(adata, "GSE131907")
