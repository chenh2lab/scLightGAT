from __future__ import annotations

from builders.caf_source import load_gse210347
from builders.common import add_data_source
from utils.config import CAF_TRAIN_TISSUES


def build(rawdir, auxdir):
    adata = load_gse210347(rawdir, auxdir)
    if "_caf_tissue_group" not in adata.obs:
        raise RuntimeError("GSE210347 canonical CAF tissue annotation is missing")

    keep = adata.obs["_caf_tissue_group"].isin(CAF_TRAIN_TISSUES)
    if not keep.any():
        raise RuntimeError(
            f"GSE210347 contains no CAF_train cells for tissues={sorted(CAF_TRAIN_TISSUES)}"
        )
    adata = adata[keep].copy()
    print(
        f"  CAF_train source-selected cells: {adata.n_obs:,} across "
        f"{sorted(adata.obs['_caf_tissue_group'].astype(str).unique())}",
        flush=True,
    )
    return add_data_source(adata, "CAF_train")
