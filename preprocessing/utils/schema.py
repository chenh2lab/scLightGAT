from __future__ import annotations

import importlib

import pandas as pd

from utils.donor import DONOR_OBS_COLUMNS, validate_donor_metadata
from utils.config import (
    CAF_TARGET_LABELS,
    MAIN_MAJOR_LABELS,
)

CORE_OBS = [
    "Data_source",
    "Celltype_training",
    "Celltype_provenance",
    "Major_resolution_source",
    "Major_source_label",
    "Celltype_coarse_source",
    "annotation_status",
    "exclusion_reason",
]

SUBTYPE_OBS = [
    "Celltype_subtraining",
    "Subtype_provenance",
    "Subtype_resolution_source",
    "Subtype_source_label",
]


def _nullable_text(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    text = values.astype(str).str.strip()
    missing_text = text.str.lower().isin({"nan", "none", "na", "n/a"})
    return values.mask(values.isna() | text.eq("") | missing_text)


def finalize_schema(adata, dataset: str, source_meta=None):
    """Normalize fields expected by the public scLightGAT code."""
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    if "Data_source" not in adata.obs:
        adata.obs["Data_source"] = dataset

    drop_columns = ["Manual_celltype", "Manual_subtype", "manual_status"]
    if dataset != "Integrated_training":
        drop_columns.extend([*SUBTYPE_OBS, "eligible_subtype"])
    adata.obs.drop(
        columns=[column for column in drop_columns if column in adata.obs],
        inplace=True,
    )
    for column in CORE_OBS:
        if column not in adata.obs:
            values = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
        else:
            values = _nullable_text(adata.obs[column])
        # Categorical columns preserve true missing values and serialize cleanly
        # in h5ad while remaining directly usable by scLightGAT/pandas.
        adata.obs[column] = pd.Categorical(values)

    if dataset == "Integrated_training":
        missing_donor = [
            column for column in DONOR_OBS_COLUMNS if column not in adata.obs
        ]
        if missing_donor:
            raise ValueError(
                "Integrated_training is missing donor metadata columns: "
                f"{missing_donor}"
            )
        for column in DONOR_OBS_COLUMNS:
            adata.obs[column] = pd.Categorical(
                _nullable_text(adata.obs[column])
            )
        validate_donor_metadata(adata.obs)

        missing_subtype = [column for column in SUBTYPE_OBS if column not in adata.obs]
        if missing_subtype:
            raise ValueError(
                "Integrated_training is missing subtype metadata columns: "
                f"{missing_subtype}"
            )
        for column in SUBTYPE_OBS:
            adata.obs[column] = pd.Categorical(_nullable_text(adata.obs[column]))
        if "eligible_subtype" not in adata.obs:
            raise ValueError("Integrated_training is missing eligible_subtype")
        adata.obs["eligible_subtype"] = (
            adata.obs["eligible_subtype"].fillna(False).astype(bool)
        )
        subtype_nonnull = adata.obs["Celltype_subtraining"].notna()
        if not adata.obs["eligible_subtype"].equals(subtype_nonnull):
            raise ValueError(
                "eligible_subtype must be true exactly when "
                "Celltype_subtraining is non-null"
            )

    if "eligible_broad" not in adata.obs:
        adata.obs["eligible_broad"] = False
    adata.obs["eligible_broad"] = (
        adata.obs["eligible_broad"].fillna(False).astype(bool)
    )
    allowed = (
        set(CAF_TARGET_LABELS)
        if dataset.startswith("CAF")
        else set(MAIN_MAJOR_LABELS)
    )
    if dataset not in {"Integrated_training", "CAF_train", "CAF_test"}:
        module = importlib.import_module(f"mapping.celltype.{dataset}")
        allowed.update(getattr(module, "EVALUATION_COARSE_LABELS", ()))
    unexpected = set(
        adata.obs["Celltype_training"].dropna().astype(str)
    ) - allowed
    if unexpected:
        raise ValueError(
            f"{dataset}: unexpected Celltype_training labels: {sorted(unexpected)}"
        )

    if adata.obs["Celltype_training"].isna().any():
        missing = int(adata.obs["Celltype_training"].isna().sum())
        raise ValueError(
            f"{dataset}: {missing} cells have no Celltype_training label"
        )

    # X is already normalized/log-transformed when final files are written.
    # Providing this layer prevents the public scLightGAT code from applying a
    # second log1p transform.
    if "log_transformed" not in adata.layers:
        adata.layers["log_transformed"] = adata.X.copy()

    adata.uns["reconstruction"] = {
        "dataset": dataset,
        "manual_annotations_applied": False,
        "label_column": "Celltype_training",
        **(source_meta or {}),
    }
    return adata
