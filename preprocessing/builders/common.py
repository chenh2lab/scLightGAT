from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import pandas as pd

from utils.downloader import download_geo, extract_archive
from utils.io import aggregate_duplicate_genes
from utils.matching import match_index


def find(root: Path, pattern: str) -> Path:
    regex = re.compile(pattern, re.IGNORECASE)
    hits = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and regex.search(path.name)
    ]
    if not hits:
        raise FileNotFoundError(f"No file under {root} matches {pattern}")
    return sorted(hits, key=lambda path: len(str(path)))[0]


def find_all(root: Path, pattern: str) -> list[Path]:
    regex = re.compile(pattern, re.IGNORECASE)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and regex.search(path.name)
    )


def best_column(frame: pd.DataFrame, patterns) -> str | None:
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for column in frame.columns:
            if regex.search(str(column)):
                return str(column)
    return None


def _metadata_id_columns(
    metadata: pd.DataFrame,
    candidates=(),
) -> list[str]:
    """Return plausible per-cell identifier columns in deterministic order.

    Dataset metadata frequently contains annotation columns such as
    ``Cell_type`` before the actual barcode/ID field.  Generic substring
    searches for ``cell`` can therefore select the wrong column.  Matching
    helpers use this function and choose the column with the largest *actual*
    overlap against expression cell IDs.
    """
    ordered: list[str] = []

    def add(column) -> None:
        if column in metadata.columns and str(column) not in ordered:
            ordered.append(str(column))

    for column in candidates:
        if column:
            add(column)

    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): str(column)
        for column in metadata.columns
    }
    for key in (
        "index",
        "cellid",
        "cellids",
        "cellname",
        "cellnames",
        "barcode",
        "barcodes",
        "cellbarcode",
        "cellbarcodes",
        "rownames",
        "rowname",
        "x",  # R write.csv/read.csv row-name column, used by GSE176078.
    ):
        column = normalized.get(key)
        if column is not None:
            add(column)

    # Last-resort name-based ID candidates.  Explicitly reject biological
    # annotation fields (cell type/subtype/cluster/label/group), which are not
    # identifiers even though their names contain the word "cell".
    annotation_words = re.compile(
        r"type|subtype|annotation|cluster|label|group|state|class|lineage",
        re.IGNORECASE,
    )
    id_words = re.compile(r"cell|barcode|(^|[_ .-])id($|[_ .-])", re.IGNORECASE)
    for column in metadata.columns:
        text = str(column)
        if id_words.search(text) and not annotation_words.search(text):
            add(column)

    return ordered


def download_geo_dataset(
    dataset_key: str,
    accession: str,
    rawdir: Path,
    patterns: dict[str, list[str]],
) -> Path:
    destination = rawdir / accession
    destination.mkdir(parents=True, exist_ok=True)
    files = download_geo(accession, patterns[dataset_key], destination)

    for path in files:
        name = path.name.lower()
        if name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
            extract_archive(path, destination / "extracted")
    return destination


def add_data_source(adata, dataset: str):
    adata.obs["Data_source"] = dataset
    return adata


def _best_metadata_match(adata, metadata: pd.DataFrame, candidates=()):
    """Find the metadata ID source with maximum real overlap to ``obs_names``."""
    best = (0, None, {})
    candidate_columns = _metadata_id_columns(metadata, candidates) + ["__index__"]

    for column in candidate_columns:
        if column == "__index__":
            ids = metadata.index.astype(str)
        else:
            ids = metadata[column].astype(str)

        pairs = match_index(ids, adata.obs_names)
        if len(pairs) > best[0]:
            best = (len(pairs), column, pairs)

    return best


def attach_metadata_by_ids(
    adata,
    metadata: pd.DataFrame,
    candidates=(),
):
    """Attach metadata to matched cells without dropping unmatched cells."""
    matched, source_column, pairs = _best_metadata_match(
        adata, metadata, candidates
    )

    if pairs:
        # Align the matched metadata rows as a DataFrame rather than filling an
        # object-typed Series cell by cell.  The previous implementation forced
        # every source column (including nCount_RNA/nFeature_RNA and numeric QC
        # fields) to dtype=object, which later makes AnnData/HDF5 serialization
        # fail when numbers and pd.NA coexist in the same object array.
        target_indices = list(pairs.keys())
        source_indices = [pairs[index] for index in target_indices]
        matched_metadata = metadata.iloc[source_indices].copy()
        matched_metadata.index = adata.obs.index[target_indices]
        aligned_metadata = matched_metadata.reindex(adata.obs.index)

        for column in aligned_metadata.columns:
            adata.obs[str(column)] = aligned_metadata[column]

    return adata, matched, source_column


def subset_to_metadata(
    adata,
    metadata: pd.DataFrame,
    candidates=(),
):
    """Keep cells that can be matched to the supplied metadata."""
    matched, source_column, pairs = _best_metadata_match(
        adata, metadata, candidates
    )
    if not pairs:
        tested = _metadata_id_columns(metadata, candidates) + ["__index__"]
        raise RuntimeError(
            "No cell IDs matched the source metadata; "
            f"tested ID sources={tested}"
        )

    target_indices = sorted(pairs)
    source_indices = [pairs[index] for index in target_indices]
    out = adata[target_indices].copy()
    matched_metadata = metadata.iloc[source_indices].copy()
    matched_metadata.index = out.obs.index

    for column in matched_metadata.columns:
        out.obs[str(column)] = matched_metadata[column].values

    return out, matched, source_column


def _aggregate_duplicate_genes(adata, label: str):
    """Make gene-symbol indices safe without discarding duplicate counts."""
    return aggregate_duplicate_genes(adata, label)


def combine_same_genes(adatas):
    cleaned = [
        _aggregate_duplicate_genes(adata, f"component {i + 1}")
        for i, adata in enumerate(adatas)
    ]
    combined = ad.concat(
        cleaned,
        join="inner",
        index_unique=None,
        merge="same",
    )
    if combined.n_vars == 0:
        raise RuntimeError(
            "Expression components have no shared gene symbols after source parsing"
        )
    return combined
