from __future__ import annotations

import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from builders.common import (
    add_data_source,
    attach_metadata_by_ids,
    combine_same_genes,
    download_geo_dataset,
    find_all,
)
from utils.config import GEO_PATTERNS, LI2018_URL
from utils.downloader import extract_archive, resumable_download


def _load_li2018_metadata(rawdir):
    destination = rawdir / "GSE123139_Li2018"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "Li2018_umi_counts.tar.gz"
    resumable_download(LI2018_URL, archive)
    extract_archive(archive, destination / "extracted")

    candidates = find_all(destination / "extracted", r"meta|annot|membership|group")
    best = None
    for path in candidates:
        try:
            frame = pd.read_csv(
                path,
                sep=None,
                engine="python",
                index_col=0,
            )
        except Exception:
            continue
        score = sum(
            re.search(r"group|annot|cell", str(column), re.IGNORECASE)
            is not None
            for column in frame.columns
        )
        if best is None or score > best[0]:
            best = (score, frame, path)

    if best is None:
        raise RuntimeError("Li2018 per-cell metadata was not found")
    return best[1]


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE123139",
        "GSE123139",
        rawdir,
        GEO_PATTERNS,
    )

    parts = []
    for path in find_all(source / "extracted", r"GSM.*\.txt(?:\.gz)?$"):
        try:
            frame = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        except Exception:
            continue
        if frame.shape[0] > frame.shape[1]:
            frame = frame.T
        parts.append(
            ad.AnnData(
                sparse.csr_matrix(frame.to_numpy(np.float32)),
                obs=pd.DataFrame(index=frame.index.astype(str)),
                var=pd.DataFrame(index=frame.columns.astype(str)),
            )
        )

    if not parts:
        raise RuntimeError("No GSE123139 per-sample matrices were parsed")

    adata = combine_same_genes(parts)
    metadata = _load_li2018_metadata(rawdir)
    metadata.index = metadata.index.astype(str)
    adata, matched, source_column = attach_metadata_by_ids(adata, metadata)
    print(
        f"  GSE123139 metadata matching: {matched:,}/{adata.n_obs:,} cells "
        f"using {source_column}"
    )
    if matched == 0:
        raise RuntimeError("Li2018 metadata did not match GSE123139 cells")

    # The thesis test cohort is the untreated MARS-seq melanoma sample set from
    # skin/subcutaneous lesions.  The GEO archive also contains unrelated GSM
    # matrices and treated/lymph-node cohorts; those are not manual-annotation
    # candidates for this benchmark.
    cohort = (
        adata.obs["type"].astype(str).eq("MARS")
        & adata.obs["treatment"].astype(str).eq("N")
        & adata.obs["location"].astype(str).isin(["(S)C", "p(S)C"])
    )
    if not cohort.any():
        raise RuntimeError("GSE123139 thesis cohort selection is empty")
    adata = adata[cohort].copy()
    adata.obs["cohort_rule"] = "MARS;treatment=N;location=(S)C|p(S)C"
    print(f"  GSE123139 thesis cohort: {adata.n_obs:,} cells", flush=True)

    return add_data_source(adata, "GSE123139")
