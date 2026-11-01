from __future__ import annotations

import re

from builders.common import (
    add_data_source,
    best_column,
    download_geo_dataset,
    find,
    find_all,
    subset_to_metadata,
)
from builders.scatomic_source import load_preserved_training_metadata
from utils.config import GEO_PATTERNS
from utils.donor import attach_donor_metadata
from utils.io import read_table, read_wide_counts


def _canonical_full_cell_id(value) -> str:
    """Normalize a complete cell ID without generating barcode-only suffix forms."""
    text = str(value).strip().lower()
    text = re.sub(r"-1$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _subset_to_lee_preserved_metadata_exact(adata, metadata):
    """Match GSE132465 only to its Lee/CRC preserved-reference rows.

    Generic suffix/barcode matching is unsafe here because 10x barcodes recur
    across patients and studies.  GSE132465 is the Lee colorectal-cancer
    dataset, and scATOMIC's preserved table contains an exact ``cell_names``
    field for these cells.  Use only unique full-ID matches.
    """
    if "dataset" not in metadata.columns:
        raise RuntimeError(
            "GSE132465 preserved scATOMIC metadata is missing the 'dataset' column"
        )

    lee = metadata["dataset"].astype(str).str.fullmatch(
        r"Lee", case=False, na=False
    )
    if not lee.any():
        raise RuntimeError(
            "GSE132465: no Lee rows were found in preserved scATOMIC metadata"
        )
    metadata = metadata.loc[lee].copy()

    if "cell_names" not in metadata.columns:
        raise RuntimeError(
            "GSE132465 preserved Lee metadata is missing 'cell_names'"
        )

    source_keys = metadata["cell_names"].map(_canonical_full_cell_id)
    target_keys = [
        _canonical_full_cell_id(value) for value in adata.obs_names.astype(str)
    ]

    # Only unique full identifiers are eligible for deterministic matching.
    source_counts = source_keys.value_counts()
    source_lookup = {
        key: i
        for i, key in enumerate(source_keys)
        if key and source_counts.get(key, 0) == 1
    }

    target_counts = {}
    for key in target_keys:
        target_counts[key] = target_counts.get(key, 0) + 1

    target_indices = []
    source_indices = []
    for ti, key in enumerate(target_keys):
        if not key or target_counts.get(key, 0) != 1:
            continue
        si = source_lookup.get(key)
        if si is None:
            continue
        target_indices.append(ti)
        source_indices.append(si)

    if not target_indices:
        raise RuntimeError(
            "GSE132465: no exact full-cell-ID matches to preserved Lee metadata"
        )

    out = adata[target_indices].copy()
    matched_metadata = metadata.iloc[source_indices].copy()
    matched_metadata.index = out.obs.index
    for column in matched_metadata.columns:
        out.obs[str(column)] = matched_metadata[column].values

    print(
        f"  GSE132465 preserved Lee metadata exact matching: "
        f"{len(target_indices):,}/{adata.n_obs:,} expression cells",
        flush=True,
    )
    return out


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE132465",
        "GSE132465",
        rawdir,
        GEO_PATTERNS,
    )
    adata = read_wide_counts(find(source, r"raw_UMI_count_matrix"))

    annotations = find_all(source, r"cell_annotation")
    if annotations:
        metadata = read_table(annotations[0], index_col=0)
        metadata.index = metadata.index.astype(str)
        sample_column = best_column(metadata, [r"sample", r"tissue", r"patient"])
        if sample_column:
            tumor = metadata[sample_column].astype(str).str.contains(
                r"-T$|tumor",
                case=False,
                regex=True,
                na=False,
            )
            if tumor.any():
                metadata = metadata.loc[tumor].copy()
        id_column = best_column(metadata, [r"cell", r"barcode"])
        adata, _, _ = subset_to_metadata(
            adata,
            metadata,
            [id_column] if id_column else (),
        )

    # scATOMIC preserved source/reference metadata are used only as source-backed
    # annotations.  GSE132465 is Lee et al. CRC; do not allow barcode-only
    # matches to rows from other studies in the combined scATOMIC table.
    preserved = load_preserved_training_metadata(rawdir, auxdir)
    adata = _subset_to_lee_preserved_metadata_exact(adata, preserved)

    adata = add_data_source(adata, "GSE132465")
    return attach_donor_metadata(adata, "GSE132465")
