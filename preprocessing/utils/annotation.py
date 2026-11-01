from __future__ import annotations

import importlib

import pandas as pd

from utils.config import (
    CAF_TARGET_LABELS,
    MAIN_MAJOR_LABELS,
)
from mapping.common import normalize_label


_MISSING = {"", "na", "nan", "none", "unknown", "undetermined", "unassigned", "<na>"}


def _usable(value) -> bool:
    return value is not None and normalize_label(value) not in _MISSING


def _mapping_name(dataset: str) -> str:
    return "CAF" if dataset.startswith("CAF_") else dataset


def _major_candidate(row, celltype_module):
    """Resolve one thesis major using dataset-specific source semantics.

    Important design rule: fine -> coarse inference belongs in each dataset's
    celltype mapper, where the source schema is known.  Do NOT run a global
    keyword resolver over every subtype/refined column.  Generic inference
    previously misread labels such as ``T/NK cells`` and ``CD4/CD8-*``.
    """
    static_major = getattr(celltype_module, "STATIC_MAJOR", None)
    if static_major is not None:
        return static_major, "STATIC_MAJOR", static_major

    # Row-based mappings can depend on several source columns.  A resolver
    # returns the exact column(s) and raw value(s) used, so the final H5AD is
    # auditable without having to infer the decision from a generic note.
    row_resolver = getattr(celltype_module, "resolve_celltype_row", None)
    if row_resolver is not None:
        mapped, column, raw = row_resolver(row)
        source = f"source:{column}" if column else "dataset-specific row mapping"
        if mapped in MAIN_MAJOR_LABELS:
            return mapped, source, raw
        return None, f"unresolved:{source}", raw

    observed = []
    for column in getattr(celltype_module, "MAJOR_COLUMNS", ()):
        if column not in row or not _usable(row[column]):
            continue
        raw = str(row[column])
        observed.append((column, raw))
        mapped = celltype_module.map_celltype(row[column])
        if mapped in MAIN_MAJOR_LABELS:
            return mapped, f"source:{column}", raw

    if observed:
        text = " | ".join(f"{c}={v}" for c, v in observed[:4])
        return None, "unresolved:source-columns", text
    return None, "unresolved:no-source-label", None


def _coarse_candidate(row, celltype_module, raw):
    mapper = getattr(celltype_module, "map_coarse_celltype_row", None)
    if mapper is not None:
        value = mapper(row)
        if _usable(value):
            return str(value)
    return str(raw) if _usable(raw) else "Unknown"


def _unresolved_status(coarse: str) -> tuple[str, str]:
    label = normalize_label(coarse)
    outside = {
        "mal",
        "malignant",
        "tumor",
        "tumorcells",
        "cancercells",
        "prostatecancercell",
        "smoothmuscle",
        "smoothmusclecells",
        "osteoclast",
        "osteoclasts",
        "pericyte",
        "pericytes",
        "glia",
        "pvl",
        "perivascularlike",
        "perivascularlikecells",
    }
    if label in outside or label.startswith("pvl") or label.endswith("pvl"):
        return "outside_ontology", "outside_ontology"
    return "unresolved", "unsupported_source_broad"



def _annotate_caf(adata, celltype):
    fine_columns = list(getattr(celltype, "FINE_COLUMNS", ()))
    targets = []
    source_columns = []
    source_labels = []
    for _, row in adata.obs.iterrows():
        selected = None
        for column in fine_columns:
            if column not in row or not _usable(row[column]):
                continue
            mapped = celltype.map_celltype(row[column])
            if mapped in CAF_TARGET_LABELS:
                selected = (mapped, column, str(row[column]))
                break
        if selected is None:
            targets.append(pd.NA)
            source_columns.append(pd.NA)
            source_labels.append(pd.NA)
        else:
            mapped, column, raw = selected
            targets.append(mapped)
            source_columns.append(f"source:{column}")
            source_labels.append(raw)

    target = pd.Series(targets, index=adata.obs.index, dtype="object")
    source = pd.Series(source_columns, index=adata.obs.index, dtype="object")
    raw = pd.Series(source_labels, index=adata.obs.index, dtype="object")
    adata.obs["Celltype_training"] = target
    adata.obs["Celltype_coarse_source"] = target.copy()
    adata.obs["Celltype_provenance"] = source
    adata.obs["Major_source_label"] = raw
    adata.obs["Major_resolution_source"] = source.copy()
    eligible = target.notna()
    adata.obs["eligible_broad"] = eligible
    adata.obs["annotation_status"] = "source_resolved"
    adata.obs.loc[~eligible, "annotation_status"] = "outside_ontology"
    adata.obs["exclusion_reason"] = pd.NA
    adata.obs.loc[~eligible, "exclusion_reason"] = "outside_ontology"
    return adata


def annotate_dataset(adata, dataset: str):
    """Resolve broad model labels from source annotations.

    Policy
    ------
    * Training labels use the thesis 13 classes; test labels may retain a
      mapper-declared source coarse class in the same ``Celltype_training``
      column.
    * Fine -> coarse inference is allowed, but only inside dataset-specific
      celltype mappers that understand that source's schema.
    * A generic cross-dataset keyword pass is never allowed to define major.
    * Incompatible/ambiguous labels are never promoted across lineages and
      never allowed to create an extra broad class.
    * Manual annotations are never consumed.
    """
    legacy_manual_columns = [
        column
        for column in ("Manual_celltype", "Manual_subtype", "manual_status")
        if column in adata.obs
    ]
    if legacy_manual_columns:
        adata.obs.drop(columns=legacy_manual_columns, inplace=True)

    name = _mapping_name(dataset)
    celltype = importlib.import_module(f"mapping.celltype.{name}")

    if dataset.startswith("CAF_"):
        return _annotate_caf(adata, celltype)

    label_values = []
    label_prov = []
    major_raw = []
    major_res_source = []
    coarse_values = []
    allowed_coarse = set(getattr(celltype, "EVALUATION_COARSE_LABELS", ()))

    for _, row in adata.obs.iterrows():
        major, source, raw = _major_candidate(row, celltype)
        resolved = major in MAIN_MAJOR_LABELS
        major_raw.append(raw if raw is not None else pd.NA)
        major_res_source.append(source)
        coarse = major if resolved else _coarse_candidate(row, celltype, raw)
        coarse_values.append(coarse)
        if resolved:
            label_values.append(major)
            label_prov.append(source)
        elif coarse in allowed_coarse:
            label_values.append(coarse)
            label_prov.append(f"source-coarse:{source}")
        else:
            label_values.append(pd.NA)
            label_prov.append(pd.NA)

    adata.obs["Celltype_training"] = pd.Series(
        label_values, index=adata.obs.index, dtype="object"
    )
    adata.obs["Celltype_provenance"] = pd.Series(
        label_prov, index=adata.obs.index, dtype="object"
    )
    adata.obs["Major_source_label"] = pd.Series(
        major_raw, index=adata.obs.index, dtype="object"
    )
    adata.obs["Major_resolution_source"] = pd.Series(
        major_res_source, index=adata.obs.index, dtype="object"
    )
    adata.obs["Celltype_coarse_source"] = pd.Series(
        coarse_values, index=adata.obs.index, dtype="object"
    )

    eligible_broad = adata.obs["Celltype_training"].notna()
    adata.obs["eligible_broad"] = eligible_broad
    adata.obs["annotation_status"] = "source_resolved"
    coarse_only = eligible_broad & ~adata.obs["Celltype_training"].isin(
        MAIN_MAJOR_LABELS
    )
    adata.obs.loc[coarse_only, "annotation_status"] = "source_coarse"
    adata.obs["exclusion_reason"] = pd.NA

    for idx in adata.obs.index[~eligible_broad]:
        status, reason = _unresolved_status(adata.obs.at[idx, "Celltype_coarse_source"])
        adata.obs.at[idx, "annotation_status"] = status
        adata.obs.at[idx, "exclusion_reason"] = reason
    return adata
