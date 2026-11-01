from __future__ import annotations

from builders.common import add_data_source
from builders.scdiffusion import download_paper_dataset, identify_dataset, load_exact_source
from utils.config import SCDIFFUSION_SAPIENS_CELLTYPES


def build(rawdir, auxdir):
    files = download_paper_dataset(rawdir, "sapiens_full")
    source_file, label_column = identify_dataset(
        files,
        expected_labels=SCDIFFUSION_SAPIENS_CELLTYPES,
        dataset_name="sapiens_full",
    )
    adata = load_exact_source(source_file, label_column)
    adata.uns["upstream_provenance"] = {
        "study": "Tabula Sapiens",
        "selection_note": "scDiffusion paper benchmark: six cell types from five organs",
    }
    return add_data_source(adata, "sapiens_full")
