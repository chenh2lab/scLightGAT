from __future__ import annotations

from builders.common import (
    add_data_source,
    best_column,
    download_geo_dataset,
    find,
    subset_to_metadata,
)
from builders.scatomic_source import (
    load_preserved_training_metadata,
    select_accession_rows,
)
from utils.config import GEO_PATTERNS
from utils.donor import attach_donor_metadata
from utils.io import read_wide_counts


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE141445",
        "GSE141445",
        rawdir,
        GEO_PATTERNS,
    )
    raw_matrix = find(
        source / "extracted",
        r"data\.raw\.matrix\.txt(?:\.gz)?$|raw.*matrix.*txt",
    )
    adata = read_wide_counts(raw_matrix)

    preserved = load_preserved_training_metadata(rawdir, auxdir)
    preserved = select_accession_rows(preserved, "GSE141445", aliases=("Chen",))
    if "disease" not in preserved.columns:
        raise RuntimeError(
            "GSE141445 preserved metadata is missing the disease field needed "
            "to distinguish Chen prostate rows from other Chen cohorts"
        )
    prostate = preserved["disease"].astype(str).str.contains(
        "prostate", case=False, na=False
    )
    preserved = preserved.loc[prostate].copy()
    if preserved.empty:
        raise RuntimeError("GSE141445 preserved metadata has no prostate-cancer rows")
    id_column = best_column(
        preserved,
        [r"cell_names?", r"barcode", r"rowname"],
    )
    adata, _, _ = subset_to_metadata(
        adata,
        preserved,
        [id_column] if id_column else (),
    )
    adata = add_data_source(adata, "GSE141445")
    return attach_donor_metadata(adata, "GSE141445")
