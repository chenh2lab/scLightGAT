from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from builders.common import add_data_source, find, find_all
from utils.config import ZHENG_ZENODO
from utils.downloader import download_zenodo_matching, extract_archive


def build(rawdir, auxdir):
    destination = rawdir / "Zheng_sorted"
    destination.mkdir(parents=True, exist_ok=True)

    archives = download_zenodo_matching(
        ZHENG_ZENODO,
        [r"scRNAseq_Benchmark_datasets\.zip$"],
        destination,
    )
    extract_archive(archives[0], destination / "extracted")

    root = destination / "extracted"
    expression_file = find(root, r"Filtered_DownSampled_SortedPBMC_data")
    label_files = find_all(root, r"Labels\.csv$")

    frame = pd.read_csv(expression_file, index_col=0, low_memory=False)
    if frame.shape[1] < frame.shape[0] and frame.shape[1] < 1000:
        frame = frame.T

    adata = ad.AnnData(
        sparse.csr_matrix(frame.to_numpy(np.float32)),
        obs=pd.DataFrame(index=frame.index.astype(str)),
        var=pd.DataFrame(index=frame.columns.astype(str)),
    )

    candidates = []
    for path in label_files:
        labels = pd.read_csv(path, low_memory=False)
        if len(labels) != adata.n_obs:
            continue
        score = int("zheng" in str(path).lower()) + int(
            path.parent == expression_file.parent
        )
        candidates.append((score, labels, path))

    if not candidates:
        raise RuntimeError("Correct Zheng_sorted Labels.csv was not found")

    _, labels, _ = max(candidates, key=lambda item: item[0])
    label_columns = [
        column
        for column in labels.columns
        if not str(column).lower().startswith("unnamed")
    ]
    if not label_columns:
        raise RuntimeError("Zheng_sorted Labels.csv has no label column")

    adata.obs["benchmark_label"] = labels[label_columns[-1]].astype(str).values
    return add_data_source(adata, "Zheng_sorted")
