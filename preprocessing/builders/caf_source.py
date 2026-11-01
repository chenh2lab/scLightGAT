from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io as spio

from builders.common import attach_metadata_by_ids, find
from utils.config import CAF_TEST_TISSUES, CAF_TRAIN_TISSUES, GEO_PATTERNS
from utils.downloader import download_geo
from utils.io import export_r_object


def _canonical_caf_tissue(value) -> str | None:
    """Map GSE210347 source tissue labels to the scLightGAT CAF split names."""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None

    aliases = (
        ("Pancreas", ("pancreas", "pancreatic", "pdac")),
        ("Bladder", ("bladder",)),
        ("Ovary", ("ovary", "ovarian")),
        ("Colon", ("colorect", "colon")),
        ("Stomach", ("stomach", "gastric")),
        ("Lung", ("lung",)),
        ("Breast", ("breast",)),
        ("Thyroid", ("thyroid",)),
        ("Prostate", ("prostate",)),
        (
            "Intrahepatic",
            ("intrahepatic", "cholangiocarcinoma", "cholangiocarcinoma", "icc"),
        ),
        ("Liver", ("liver", "hepatocellular", "hcc")),
    )
    for canonical, tokens in aliases:
        if any(token in text for token in tokens):
            return canonical
    return None


def _source_column(metadata: pd.DataFrame, wanted: str) -> str:
    matches = [column for column in metadata.columns if str(column).lower() == wanted]
    if not matches:
        raise RuntimeError(
            f"GSE210347 metadata is missing required '{wanted}' column; "
            f"available={list(metadata.columns)}"
        )
    return str(matches[0])


def load_gse210347(rawdir: Path, auxdir: Path):
    destination = rawdir / "GSE210347"
    destination.mkdir(parents=True, exist_ok=True)
    download_geo("GSE210347", GEO_PATTERNS["CAF"], destination)

    counts = find(destination, r"counts\.Rds\.gz$")
    metadata_file = find(destination, r"meta\.txt\.gz$")

    # GSE210347 publishes 855,271 cells.  Exporting the entire sparse matrix to
    # MatrixMarket is unnecessarily expensive because scLightGAT's CAF task uses
    # only fibroblasts from its train/test tissue split.  The public metadata is
    # source-backed and contains cellname, celltype, tissue and cluster, so use it
    # to select model-relevant columns before R export.
    metadata = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    cell_id_column = _source_column(metadata, "cellname")
    celltype_column = _source_column(metadata, "celltype")
    tissue_column = _source_column(metadata, "tissue")

    fibroblast = metadata[celltype_column].astype(str).str.contains(
        r"fibro|\bcaf\b",
        case=False,
        regex=True,
        na=False,
    )
    metadata = metadata.loc[fibroblast].copy()
    if metadata.empty:
        raise RuntimeError("GSE210347 metadata contains no source-annotated fibroblast/CAF cells")

    metadata["_caf_tissue_group"] = metadata[tissue_column].map(_canonical_caf_tissue)
    wanted_tissues = set(CAF_TRAIN_TISSUES) | set(CAF_TEST_TISSUES)
    relevant = metadata["_caf_tissue_group"].isin(wanted_tissues)
    selected_metadata = metadata.loc[relevant].copy()
    if selected_metadata.empty:
        examples = metadata[tissue_column].astype(str).value_counts().head(30).to_dict()
        raise RuntimeError(
            "GSE210347 fibroblast metadata did not match any configured CAF tissues; "
            f"source tissue examples={examples}"
        )

    selected_ids = selected_metadata[cell_id_column].astype(str)
    duplicate_ids = int(selected_ids.duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(
            f"GSE210347 metadata has {duplicate_ids:,} duplicate cellname values in the CAF subset"
        )

    counts_by_tissue = (
        selected_metadata["_caf_tissue_group"].value_counts().sort_index().to_dict()
    )
    print(
        "  GSE210347 CAF source selection: "
        f"{len(selected_metadata):,} fibroblast cells; tissues={counts_by_tissue}",
        flush=True,
    )

    selected_ids_file = auxdir / "GSE210347_selected_CAF_cell_ids.txt"
    selected_ids_file.parent.mkdir(parents=True, exist_ok=True)
    selected_ids_file.write_text("\n".join(selected_ids.tolist()) + "\n")

    export_dir = auxdir / "GSE210347_counts_export"
    export_r_object(counts, export_dir, selected_columns=selected_ids_file)

    matrix_file = find(export_dir, r"\.mtx$")
    row_file = find(export_dir, r"\.rows\.txt$")
    column_file = find(export_dir, r"\.cols\.txt$")

    matrix = spio.mmread(matrix_file).tocsr()
    row_names = Path(row_file).read_text().splitlines()
    column_names = Path(column_file).read_text().splitlines()

    if matrix.shape != (len(row_names), len(column_names)):
        raise RuntimeError(
            "GSE210347 exported matrix dimensions are inconsistent: "
            f"matrix={matrix.shape}, rows={len(row_names)}, cols={len(column_names)}"
        )

    # The published R object is genes x cells.
    adata = ad.AnnData(
        matrix.T.tocsr(),
        obs=pd.DataFrame(index=column_names),
        var=pd.DataFrame(index=row_names),
    )

    # Attach only the same source-backed metadata rows used for R column
    # selection.  Exact cellname matching is expected here.
    adata, matched, source = attach_metadata_by_ids(
        adata,
        selected_metadata,
        [cell_id_column],
    )
    if matched != adata.n_obs:
        raise RuntimeError(
            "GSE210347 metadata/expression mismatch after source selection: "
            f"matched={matched:,}, expression_cells={adata.n_obs:,}, source={source}"
        )
    return adata
