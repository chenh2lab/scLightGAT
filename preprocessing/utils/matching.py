from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd


def canonical_cell_id(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"-1$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def candidate_cell_id_forms(value) -> set[str]:
    text = str(value).strip()
    candidates = {canonical_cell_id(text)}
    tokens = [
        token
        for token in re.split(r"[\s_:|/\\.\-]+", text)
        if token
    ]

    for width in (1, 2, 3):
        if len(tokens) >= width:
            candidates.add(canonical_cell_id("_".join(tokens[-width:])))

    if len(tokens) >= 2:
        candidates.add(canonical_cell_id(f"{tokens[0]}_{tokens[-1]}"))

    return {candidate for candidate in candidates if len(candidate) >= 6}


def match_index(source_ids, target_ids) -> dict[int, int]:
    """Return target-index -> source-index for unique normalized cell IDs."""
    target_forms: dict[str, list[int]] = defaultdict(list)
    for target_index, cell_id in enumerate(map(str, target_ids)):
        for form in candidate_cell_id_forms(cell_id):
            target_forms[form].append(target_index)

    unique_target_forms = {
        form: indices[0]
        for form, indices in target_forms.items()
        if len(indices) == 1
    }

    pairs: dict[int, int] = {}
    for source_index, cell_id in enumerate(map(str, source_ids)):
        for form in candidate_cell_id_forms(cell_id):
            target_index = unique_target_forms.get(form)
            if target_index is not None and target_index not in pairs:
                pairs[target_index] = source_index
                break

    return pairs


def attach_metadata(adata, metadata, id_column=None, prefix="source_"):
    """Attach all metadata columns to matching AnnData cells."""
    if id_column and id_column in metadata:
        source_ids = metadata[id_column].astype(str)
    else:
        source_ids = metadata.index.astype(str)

    pairs = match_index(source_ids, adata.obs_names)
    for column in metadata.columns:
        values = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
        for target_index, source_index in pairs.items():
            values.iloc[target_index] = metadata.iloc[source_index][column]
        adata.obs[f"{prefix}{column}"] = values.values

    return adata, len(pairs)
