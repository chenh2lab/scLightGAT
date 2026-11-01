from __future__ import annotations

import pandas as pd

from builders.common import (
    add_data_source,
    attach_metadata_by_ids,
    download_geo_dataset,
    find,
)
from utils.config import GEO_PATTERNS, GSE153935_ZENODO
from utils.downloader import download_zenodo_matching
from utils.io import export_r_object, read_wide_counts
from utils.matching import match_index


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE153935",
        "GSE153935",
        rawdir,
        GEO_PATTERNS,
    )
    adata = read_wide_counts(find(source, r"TLDS_AllCells"))

    zenodo_dir = rawdir / "GSE153935_zenodo"
    zenodo_dir.mkdir(parents=True, exist_ok=True)
    files = download_zenodo_matching(
        GSE153935_ZENODO,
        [r"TLDS_AllCells_RPCAintegrated\.Rdata$"],
        zenodo_dir,
    )

    export_dir = auxdir / "GSE153935_rmeta"
    export_r_object(files[0], export_dir, require_matrix=False)

    best = None
    for path in export_dir.rglob("*.csv"):
        metadata = pd.read_csv(path, low_memory=False)
        if "__rowname__" in metadata:
            ids = metadata["__rowname__"].astype(str)
        else:
            ids = metadata.index.astype(str)
        overlap = len(match_index(ids, adata.obs_names))
        if best is None or overlap > best[0]:
            best = (overlap, metadata, path)

    if best is None or best[0] == 0:
        raise RuntimeError("GSE153935 original-study metadata did not match cells")

    adata, _, _ = attach_metadata_by_ids(
        adata,
        best[1],
        ["__rowname__"],
    )
    return add_data_source(adata, "GSE153935")
