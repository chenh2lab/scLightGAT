from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from builders.common import (
    add_data_source,
    attach_metadata_by_ids,
    best_column,
    combine_same_genes,
    download_geo_dataset,
    find,
    find_all,
)
from utils.config import GEO_PATTERNS


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE166555",
        "GSE166555",
        rawdir,
        GEO_PATTERNS,
    )

    parts = []
    for path in find_all(source / "extracted", r"\.tsv(?:\.gz)?$"):
        if "meta" in path.name.lower():
            continue
        try:
            frame = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        except Exception:
            continue
        if frame.shape[0] > frame.shape[1]:
            frame = frame.T
        if frame.shape[1] < 100:
            continue
        parts.append(
            ad.AnnData(
                sparse.csr_matrix(frame.to_numpy(np.float32)),
                obs=pd.DataFrame(index=frame.index.astype(str)),
                var=pd.DataFrame(index=frame.columns.astype(str)),
            )
        )

    if not parts:
        raise RuntimeError("GSE166555 count matrices were not parsed")

    adata = combine_same_genes(parts)
    metadata = pd.read_csv(
        find(source, r"meta_data"),
        sep="\t",
        index_col=0,
        low_memory=False,
    )
    metadata.index = metadata.index.astype(str)
    adata, matched, source_column = attach_metadata_by_ids(adata, metadata)
    if matched == 0:
        raise RuntimeError("GSE166555 metadata did not match expression cells")
    print(
        f"  GSE166555 metadata matching: {matched:,}/{len(metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )

    condition_column = best_column(
        adata.obs,
        [r"condition", r"tissue", r"origin"],
    )
    if condition_column:
        tumor = adata.obs[condition_column].astype(str).str.contains(
            "tumor",
            case=False,
            na=False,
        )
        if tumor.sum() > 1000:
            adata = adata[tumor].copy()

    return add_data_source(adata, "GSE166555")
