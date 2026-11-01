from __future__ import annotations

import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from mapping.common import normalize_label
from utils.donor import DONOR_OBS_COLUMNS, validate_donor_metadata
from utils.io import atomic_write_h5ad


SUBTYPE_ASSIGNMENT_SCHEMA_VERSION = 1
SUBTYPE_ASSIGNMENT_PATH = (
    Path(__file__).resolve().parent.parent / "subtype_donor_split.json"
)
SUBTYPE_SPLIT_STRATEGY = "fixed per-broad donor-held-out assignment"
SUBTYPE_SPLIT_ALGORITHM_VERSION = 3
MIN_DONORS_PER_SIDE_PER_SUBTYPE = 2

# Canonical subtype ontology retained from the source-backed annotations in the
# original integrated training file.  The values are the only supported parent
# broad labels for the subtype task.
SUBTYPES_BY_BROAD = {
    "B cells": (
        "Follicular B cells",
        "Germinal B cells",
        "MALT B cells",
        "Memory B cells",
        "Naive B cells",
    ),
    "CD4+T cells": (
        "CD4+Tfh/Th cells",
        "CD4+exhausted T cells",
        "CD4+memory T cells",
        "CD4+naive T cells",
        "CD4+reg T cells",
    ),
    "CD8+T cells": (
        "CD8+MAIT T cells",
        "CD8+Naive T cells",
        "CD8+exhausted T cells",
        "CD8+memory T cells",
    ),
    "DC": (
        "DC",
        "cDC",
    ),
    "Plasma cells": (
        "IgA+ Plasma",
        "IgG+ Plasma",
        "Plasma cells",
        "Plasmablasts",
    ),
}

SUBTYPE_OUTPUT_STEMS = {
    "B cells": "B_cell",
    "CD4+T cells": "CD4_T_cell",
    "CD8+T cells": "CD8_T_cell",
    "DC": "DC",
    "Plasma cells": "Plasma_cell",
}

SUBTYPE_OBS_COLUMNS = (
    "Celltype_subtraining",
    "Subtype_provenance",
    "Subtype_resolution_source",
    "Subtype_source_label",
    "eligible_subtype",
)

_SUBTYPE_PARENT = {
    subtype: broad
    for broad, subtypes in SUBTYPES_BY_BROAD.items()
    for subtype in subtypes
}


def _normalized_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {normalize_label(source): target for source, target in mapping.items()}


# These are exact, dataset-specific translations of author-provided metadata.
# They deliberately avoid a generic marker/keyword inference pass.
_SOURCE_SUBTYPE_SPECS = {
    "GSE131907": {
        "columns": ("Cell_subtype", "cell_subtype"),
        "mapping": _normalized_mapping(
            {
                "Follicular B cells": "Follicular B cells",
                "GC B cells in the DZ": "Germinal B cells",
                "GC B cells in the LZ": "Germinal B cells",
                "MALT B cells": "MALT B cells",
                "CD4+ Th": "CD4+Tfh/Th cells",
                "Exhausted Tfh": "CD4+exhausted T cells",
                "Naive CD4+ T": "CD4+naive T cells",
                "Treg": "CD4+reg T cells",
                "Exhausted CD8+ T": "CD8+exhausted T cells",
                "Naive CD8+ T": "CD8+Naive T cells",
                "Activated DCs": "DC",
                "CD141+ DCs": "cDC",
                "CD163+CD14+ DCs": "DC",
                "CD1c+ DCs": "cDC",
                "Plasma cells": "Plasma cells",
            }
        ),
    },
    "GSE132465": {
        "columns": ("original_annotation.2",),
        "mapping": _normalized_mapping(
            {
                "T follicular helper cells": "CD4+Tfh/Th cells",
                "T helper 17 cells": "CD4+Tfh/Th cells",
                "cDC": "cDC",
                "IgA+ Plasma": "IgA+ Plasma",
                "IgG+ Plasma": "IgG+ Plasma",
            }
        ),
    },
    "GSE140228": {
        "columns": ("celltype_sub",),
        "mapping": _normalized_mapping(
            {
                "CD4-C3-ANXA1": "CD4+memory T cells",
                "CD4-C4-IL7R": "CD4+memory T cells",
                "CD4-C5-TCF7": "CD4+naive T cells",
                "CD4-C6-CXCL13": "CD4+Tfh/Th cells",
                "CD4-C7-FOXP3": "CD4+reg T cells",
                "CD8-C3-IL7R": "CD8+memory T cells",
                "CD8-C4-CX3CR1": "CD8+memory T cells",
                "CD8-C5-SELL": "CD8+Naive T cells",
                "CD8-C6-GZMK": "CD8+memory T cells",
                "CD8-C7-KLRD1": "CD8+memory T cells",
                "CD8-C8-PDCD1": "CD8+exhausted T cells",
                "CD8-C9-SLC4A10": "CD8+MAIT T cells",
                "DC-C1-CD1C": "cDC",
                "DC-C2-FCER1A": "cDC",
                "DC-C3-CLEC9A": "cDC",
                "DC-C4-LAMP3": "DC",
                "Lymphoid-B-Plasma": "Plasma cells",
            }
        ),
    },
    "GSE176078": {
        "columns": ("celltype_subset",),
        "mapping": _normalized_mapping(
            {
                "B cells Memory": "Memory B cells",
                "B cells Naive": "Naive B cells",
                "T_cells_c0_CD4+_CCR7": "CD4+naive T cells",
                "T_cells_c1_CD4+_IL7R": "CD4+memory T cells",
                "T_cells_c2_CD4+_T-regs_FOXP3": "CD4+reg T cells",
                "T_cells_c3_CD4+_Tfh_CXCL13": "CD4+Tfh/Th cells",
                "T_cells_c5_CD8+_GZMK": "CD8+memory T cells",
                "T_cells_c8_CD8+_LAG3": "CD8+exhausted T cells",
                "Myeloid_c0_DC_LAMP3": "DC",
                "Myeloid_c11_cDC2_CD1C": "cDC",
                "Myeloid_c3_cDC1_CLEC9A": "cDC",
                "Plasmablasts": "Plasmablasts",
            }
        ),
    },
}

_MISSING_SOURCE_LABELS = {
    "",
    "<na>",
    "na",
    "n/a",
    "nan",
    "none",
    "unassigned",
    "undetermined",
    "unknown",
}


def _source_values(
    obs: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    labels = pd.Series(pd.NA, index=obs.index, dtype="object")
    sources = pd.Series(pd.NA, index=obs.index, dtype="object")

    for column in columns:
        if column not in obs:
            continue
        values = obs[column].astype("object")
        text = values.astype(str).str.strip()
        usable = (
            labels.isna()
            & values.notna()
            & ~text.str.lower().isin(_MISSING_SOURCE_LABELS)
        )
        labels.loc[usable] = text.loc[usable]
        sources.loc[usable] = column
    return labels, sources


def attach_source_subtypes(adata, dataset: str):
    """Attach canonical subtype labels from exact author metadata mappings."""
    spec = _SOURCE_SUBTYPE_SPECS.get(dataset)
    if spec is None:
        source_labels = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
        source_columns = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
        mapped = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
        resolution = pd.Series(
            "unavailable:no-subtype-mapping",
            index=adata.obs.index,
            dtype="object",
        )
    else:
        source_labels, source_columns = _source_values(
            adata.obs,
            spec["columns"],
        )
        mapped = source_labels.map(
            lambda value: (
                spec["mapping"].get(normalize_label(value), pd.NA)
                if pd.notna(value)
                else pd.NA
            )
        ).astype("object")
        resolution = pd.Series(
            "unresolved:no-source-subtype",
            index=adata.obs.index,
            dtype="object",
        )
        has_source = source_labels.notna()
        resolution.loc[has_source] = (
            "unresolved:source:" + source_columns.loc[has_source].astype(str)
        )
        resolved = mapped.notna()
        resolution.loc[resolved] = (
            "source:" + source_columns.loc[resolved].astype(str)
        )

    resolved = mapped.notna()
    if resolved.any():
        # Compare only rows with a resolved subtype.  Comparing the complete
        # object Series evaluates ``pd.NA != pd.NA`` for unresolved rows before
        # the ``resolved`` mask can be applied, which raises because the boolean
        # value of ``pd.NA`` is undefined.
        resolved_index = mapped.index[resolved]
        broad = adata.obs.loc[resolved_index, "Celltype_training"].astype("string")
        expected_parent = mapped.loc[resolved_index].map(_SUBTYPE_PARENT).astype(
            "string"
        )
        mismatch = broad.ne(expected_parent).fillna(True)
        if mismatch.any():
            mismatch_index = mismatch.index[mismatch]
            examples = pd.DataFrame(
                {
                    "broad": broad.loc[mismatch_index],
                    "subtype": mapped.loc[mismatch_index],
                    "source_label": source_labels.loc[mismatch_index],
                }
            ).head(5)
            raise ValueError(
                f"{dataset}: source subtype conflicts with its broad parent; "
                f"examples={examples.to_dict(orient='records')}"
            )

    provenance = pd.Series(pd.NA, index=adata.obs.index, dtype="object")
    provenance.loc[resolved] = resolution.loc[resolved]
    adata.obs["Celltype_subtraining"] = mapped
    adata.obs["Subtype_provenance"] = provenance
    adata.obs["Subtype_resolution_source"] = resolution
    adata.obs["Subtype_source_label"] = source_labels
    adata.obs["eligible_subtype"] = resolved.astype(bool)
    return adata


def validate_subtype_obs(obs: pd.DataFrame) -> None:
    required = {"Celltype_training", "Celltype_subtraining"}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(
            "Subtype split requires these obs columns: "
            f"{sorted(required)}; missing={sorted(missing)}"
        )

    subtype = obs["Celltype_subtraining"].astype("object")
    broad = obs["Celltype_training"].astype("object")
    resolved = subtype.notna()
    if not resolved.any():
        raise ValueError("No non-null Celltype_subtraining labels are available")

    unexpected = set(subtype.loc[resolved].astype(str)) - set(_SUBTYPE_PARENT)
    if unexpected:
        raise ValueError(f"Unexpected subtype labels: {sorted(unexpected)}")

    resolved_index = subtype.index[resolved]
    resolved_broad = broad.loc[resolved_index].astype("string")
    expected_parent = subtype.loc[resolved_index].map(_SUBTYPE_PARENT).astype(
        "string"
    )
    mismatch = resolved_broad.ne(expected_parent).fillna(True)
    if mismatch.any():
        mismatch_index = mismatch.index[mismatch]
        examples = pd.DataFrame(
            {
                "broad": resolved_broad.loc[mismatch_index],
                "subtype": subtype.loc[mismatch_index],
            }
        ).head(5)
        raise ValueError(
            "Subtype labels conflict with their broad parents; "
            f"examples={examples.to_dict(orient='records')}"
        )

    if "eligible_subtype" in obs:
        eligible = obs["eligible_subtype"].fillna(False).astype(bool)
        if not eligible.equals(resolved.astype(bool)):
            raise ValueError(
                "eligible_subtype must be true exactly when "
                "Celltype_subtraining is non-null"
            )


def subtype_output_paths(output_dir: Path) -> dict[tuple[str, str], Path]:
    output_dir = Path(output_dir)
    return {
        (broad, split): output_dir / f"{stem}_{split}.h5ad"
        for broad, stem in SUBTYPE_OUTPUT_STEMS.items()
        for split in ("train", "test")
    }


def load_fixed_donor_assignment(
    path: Path = SUBTYPE_ASSIGNMENT_PATH,
) -> pd.DataFrame:
    """Load and strictly validate the authoritative donor split JSON."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fixed subtype donor assignment is missing: {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Fixed subtype donor assignment must be a JSON object")
    required_root_keys = {
        "schema_version",
        "strategy",
        "scope",
        "group_id_format",
        "broad_assignments",
    }
    if set(payload) != required_root_keys:
        raise ValueError(
            "Fixed subtype donor assignment root keys differ from the schema; "
            f"observed={sorted(payload)}, expected={sorted(required_root_keys)}"
        )
    if int(payload.get("schema_version", -1)) != SUBTYPE_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported fixed subtype donor assignment schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("strategy") != SUBTYPE_SPLIT_STRATEGY:
        raise ValueError(
            "Fixed subtype donor assignment strategy does not match the "
            f"production strategy: {payload.get('strategy')!r}"
        )
    if payload.get("scope") != "within_broad_across_subtypes":
        raise ValueError(
            "Fixed subtype donor assignment scope must be "
            "'within_broad_across_subtypes'"
        )
    if payload.get("group_id_format") != "Data_source:Donor_ID":
        raise ValueError(
            "Fixed subtype donor assignment group_id_format must be "
            "'Data_source:Donor_ID'"
        )

    configured = payload.get("broad_assignments")
    if not isinstance(configured, dict):
        raise ValueError("broad_assignments must be a JSON object")
    if set(configured) != set(SUBTYPES_BY_BROAD):
        raise ValueError(
            "Fixed subtype donor assignment broad keys differ from the "
            f"canonical tasks; configured={sorted(configured)}, "
            f"expected={sorted(SUBTYPES_BY_BROAD)}"
        )

    rows: list[dict[str, str]] = []
    for broad in SUBTYPES_BY_BROAD:
        split_lists = configured[broad]
        if not isinstance(split_lists, dict) or set(split_lists) != {
            "train",
            "test",
        }:
            raise ValueError(
                f"{broad}: assignment must contain exactly train and test lists"
            )

        seen: set[str] = set()
        for split in ("train", "test"):
            groups = split_lists[split]
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"{broad}/{split}: donor list must be non-empty")
            if any(not isinstance(group, str) or group.strip() != group for group in groups):
                raise ValueError(
                    f"{broad}/{split}: every Group_ID must be a trimmed string"
                )
            if groups != sorted(groups):
                raise ValueError(f"{broad}/{split}: Group_ID list must be sorted")
            if len(groups) != len(set(groups)):
                raise ValueError(f"{broad}/{split}: duplicate Group_ID values")
            overlap = seen.intersection(groups)
            if overlap:
                raise ValueError(
                    f"{broad}: Group_ID values occur in both train and test: "
                    f"{sorted(overlap)}"
                )
            seen.update(groups)

            for group in groups:
                if ":" not in group:
                    raise ValueError(
                        f"{broad}/{split}: invalid namespaced Group_ID {group!r}"
                    )
                data_source, donor_id = group.split(":", 1)
                if not data_source or not donor_id:
                    raise ValueError(
                        f"{broad}/{split}: invalid namespaced Group_ID {group!r}"
                    )
                rows.append(
                    {
                        "broad_type": broad,
                        "Group_ID": group,
                        "Data_source": data_source,
                        "split": split,
                    }
                )

    assignment = pd.DataFrame(rows)
    if assignment.duplicated(["broad_type", "Group_ID"]).any():
        raise ValueError("Fixed subtype donor assignment contains duplicate keys")
    return assignment


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(value: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def subtype_outputs_complete(
    output_dir: Path,
    *,
    assignment_path: Path = SUBTYPE_ASSIGNMENT_PATH,
) -> bool:
    # The JSON is the source of truth.  A malformed production assignment is
    # a configuration error and must not be mistaken for merely stale output.
    fixed_assignment = load_fixed_donor_assignment(assignment_path)
    assignment_file = Path(assignment_path).name
    output_dir = Path(output_dir)
    paths = subtype_output_paths(output_dir)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return False

    assignment_csv_path = output_dir / "subtype_donor_assignments.csv"
    manifest_path = output_dir / "subtype_split_manifest.json"
    summary_path = output_dir / "subtype_split_summary.csv"
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in (assignment_csv_path, manifest_path, summary_path)
    ):
        return False

    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("strategy") != SUBTYPE_SPLIT_STRATEGY:
            return False
        if manifest.get("scope") != "within_broad_across_subtypes":
            return False
        if int(manifest.get("algorithm_version", -1)) != SUBTYPE_SPLIT_ALGORITHM_VERSION:
            return False
        if int(manifest.get("assignment_schema_version", -1)) != (
            SUBTYPE_ASSIGNMENT_SCHEMA_VERSION
        ):
            return False
        if manifest.get("assignment_file") != assignment_file:
            return False

        assignment = pd.read_csv(assignment_csv_path, dtype=str)
        required_assignment = {
            "broad_type",
            "Group_ID",
            "Data_source",
            "split",
            "n_cells",
            "n_subtypes",
        }
        if required_assignment.difference(assignment.columns):
            return False
        if assignment.duplicated(["broad_type", "Group_ID"]).any():
            return False
        if set(assignment["split"]) != {"train", "test"}:
            return False
        fixed_records = {
            (row.broad_type, row.Group_ID): (row.Data_source, row.split)
            for row in fixed_assignment.itertuples(index=False)
        }
        output_records = {
            (row.broad_type, row.Group_ID): (row.Data_source, row.split)
            for row in assignment.itertuples(index=False)
        }
        if output_records != fixed_records:
            return False
        source_split_coverage = assignment.groupby(
            ["broad_type", "Data_source"],
            observed=True,
        )["split"].agg(lambda values: set(values))
        if any(
            splits != {"train", "test"}
            for splits in source_split_coverage
        ):
            return False
        if int(manifest.get("n_broad_donor_assignments", -1)) != len(assignment):
            return False
        if int(manifest.get("n_train_assignments", -1)) != int(
            assignment["split"].eq("train").sum()
        ):
            return False
        if int(manifest.get("n_test_assignments", -1)) != int(
            assignment["split"].eq("test").sum()
        ):
            return False

        expected_summary_keys = set(paths)
        summary = pd.read_csv(summary_path, dtype=str)
        required_summary = {
            "broad_type",
            "split",
            "n_cells",
            "n_groups",
            "n_genes",
            "output",
        }
        if required_summary.difference(summary.columns):
            return False
        if summary.duplicated(["broad_type", "split"]).any():
            return False
        observed_summary_keys = set(
            zip(summary["broad_type"], summary["split"])
        )
        if observed_summary_keys != expected_summary_keys:
            return False
        summary = summary.set_index(["broad_type", "split"])

        expected_manifest_files = [str(path.name) for path in paths.values()]
        if manifest.get("files") != expected_manifest_files:
            return False
        manifest_broad_stats = manifest.get("broad_stats", {})
        if set(manifest_broad_stats) != set(SUBTYPES_BY_BROAD):
            return False

        observed_group_records: dict[
            tuple[str, str], tuple[str, str, int, int]
        ] = {}
        observed_cell_ids: set[str] = set()
        required_layers = {"counts", "log_transformed", "raw_counts"}
        reference_var_names: pd.Index | None = None
        reference_raw_var_names: pd.Index | None = None
        for broad, expected_subtypes in SUBTYPES_BY_BROAD.items():
            broad_assignment = assignment.loc[
                assignment["broad_type"].eq(broad)
            ]
            expected_broad_stats = {
                "n_groups": len(broad_assignment),
                "n_train_groups": int(
                    broad_assignment["split"].eq("train").sum()
                ),
                "n_test_groups": int(
                    broad_assignment["split"].eq("test").sum()
                ),
                "n_eligible_cells": int(
                    broad_assignment["n_cells"].astype(int).sum()
                ),
            }
            manifest_stats = manifest_broad_stats[broad]
            if any(
                int(manifest_stats.get(field, -1)) != expected
                for field, expected in expected_broad_stats.items()
            ):
                return False
            summaries = {}
            for split in ("train", "test"):
                source = ad.read_h5ad(paths[(broad, split)], backed="r")
                try:
                    validate_donor_metadata(source.obs)
                    observed_broad = set(
                        source.obs["Celltype_training"].astype(str)
                    )
                    observed_subtypes = set(
                        source.obs["Celltype_subtraining"].astype(str)
                    )
                    if observed_broad != {broad}:
                        return False
                    if observed_subtypes != set(expected_subtypes):
                        return False
                    metadata = source.uns.get("subtype_split", {})
                    if metadata.get("broad_type") != broad:
                        return False
                    if metadata.get("split") != split:
                        return False
                    if metadata.get("strategy") != SUBTYPE_SPLIT_STRATEGY:
                        return False
                    if int(metadata.get("algorithm_version", -1)) != SUBTYPE_SPLIT_ALGORITHM_VERSION:
                        return False
                    if int(metadata.get("assignment_schema_version", -1)) != (
                        SUBTYPE_ASSIGNMENT_SCHEMA_VERSION
                    ):
                        return False
                    if metadata.get("assignment_file") != assignment_file:
                        return False
                    if metadata.get("group_column") != "Group_ID":
                        return False
                    if metadata.get("scope") != "within_broad_across_subtypes":
                        return False
                    metadata_broad_stats = {
                        "n_groups_broad": expected_broad_stats["n_groups"],
                        "n_train_groups_broad": expected_broad_stats[
                            "n_train_groups"
                        ],
                        "n_test_groups_broad": expected_broad_stats[
                            "n_test_groups"
                        ],
                        "n_eligible_cells_broad": expected_broad_stats[
                            "n_eligible_cells"
                        ],
                    }
                    if any(
                        int(metadata.get(field, -1)) != expected
                        for field, expected in metadata_broad_stats.items()
                    ):
                        return False
                    if source.raw is None:
                        return False
                    layer_names = set(source.layers.keys())
                    if not required_layers.issubset(layer_names):
                        return False
                    var_names = pd.Index(source.var_names.astype(str))
                    raw_var_names = pd.Index(source.raw.var_names.astype(str))
                    if reference_var_names is None:
                        reference_var_names = var_names
                        reference_raw_var_names = raw_var_names
                    elif not reference_var_names.equals(var_names):
                        return False
                    elif not reference_raw_var_names.equals(raw_var_names):
                        return False
                    groups = set(source.obs["Group_ID"].astype(str))
                    cell_ids = set(source.obs_names.astype(str))
                    summary_row = summary.loc[(broad, split)]
                    if int(summary_row["n_cells"]) != int(source.n_obs):
                        return False
                    if int(summary_row["n_groups"]) != len(groups):
                        return False
                    if int(metadata.get("n_groups_in_file", -1)) != len(groups):
                        return False
                    if int(summary_row["n_genes"]) != int(source.n_vars):
                        return False
                    recorded_output = Path(summary_row["output"]).expanduser()
                    if recorded_output.resolve() != paths[(broad, split)].resolve():
                        return False
                    if observed_cell_ids & cell_ids:
                        return False
                    observed_cell_ids.update(cell_ids)
                    group_details = source.obs.groupby(
                        "Group_ID",
                        observed=True,
                    ).agg(
                        Data_source=("Data_source", "first"),
                        n_cells=("Group_ID", "size"),
                        n_subtypes=("Celltype_subtraining", "nunique"),
                    )
                    for group, details in group_details.iterrows():
                        key = (broad, group)
                        if key in observed_group_records:
                            return False
                        observed_group_records[key] = (
                            str(details["Data_source"]),
                            split,
                            int(details["n_cells"]),
                            int(details["n_subtypes"]),
                        )
                    summaries[split] = {
                        "ids": set(source.obs_names.astype(str)),
                        "groups": groups,
                        "counts": source.obs[
                            "Celltype_subtraining"
                        ].astype(str).value_counts(),
                        "donor_counts": source.obs.groupby(
                            "Celltype_subtraining",
                            observed=True,
                        )["Group_ID"].nunique(),
                        "n_vars": source.n_vars,
                        "var_names": var_names,
                        "raw_var_names": raw_var_names,
                        "layers": layer_names,
                    }
                finally:
                    source.file.close()

            if summaries["train"]["ids"] & summaries["test"]["ids"]:
                return False
            if summaries["train"]["groups"] & summaries["test"]["groups"]:
                return False
            if summaries["train"]["n_vars"] != summaries["test"]["n_vars"]:
                return False
            if not summaries["train"]["var_names"].equals(
                summaries["test"]["var_names"]
            ):
                return False
            if not summaries["train"]["raw_var_names"].equals(
                summaries["test"]["raw_var_names"]
            ):
                return False
            if summaries["train"]["layers"] != summaries["test"]["layers"]:
                return False

            train_counts = summaries["train"]["counts"]
            test_counts = summaries["test"]["counts"]
            for subtype in expected_subtypes:
                if int(train_counts.get(subtype, 0)) <= 0:
                    return False
                if int(test_counts.get(subtype, 0)) <= 0:
                    return False
                if int(
                    summaries["train"]["donor_counts"].get(subtype, 0)
                ) < MIN_DONORS_PER_SIDE_PER_SUBTYPE:
                    return False
                if int(
                    summaries["test"]["donor_counts"].get(subtype, 0)
                ) < MIN_DONORS_PER_SIDE_PER_SUBTYPE:
                    return False

        expected_group_records = {
            (row.broad_type, row.Group_ID): (
                row.Data_source,
                row.split,
                int(row.n_cells),
                int(row.n_subtypes),
            )
            for row in assignment.itertuples(index=False)
        }
        if observed_group_records != expected_group_records:
            return False
        if len(observed_cell_ids) != int(manifest.get("n_eligible_cells", -1)):
            return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def plan_donor_subtype_split(
    obs: pd.DataFrame,
    *,
    assignment_path: Path = SUBTYPE_ASSIGNMENT_PATH,
) -> dict:
    validate_subtype_obs(obs)
    validate_donor_metadata(obs)
    fixed_assignment = load_fixed_donor_assignment(assignment_path)
    broad_values = obs["Celltype_training"].astype("object")
    subtype_values = obs["Celltype_subtraining"].astype("object")
    eligible_mask = subtype_values.notna().to_numpy()
    eligible = obs.loc[eligible_mask].copy()
    eligible["_class"] = (
        eligible["Celltype_training"].astype(str)
        + "\x1f"
        + eligible["Celltype_subtraining"].astype(str)
    )

    expected_classes = [
        f"{broad}\x1f{subtype}"
        for broad, subtypes in SUBTYPES_BY_BROAD.items()
        for subtype in subtypes
    ]
    observed_classes = set(eligible["_class"])
    missing_classes = set(expected_classes) - observed_classes
    unexpected_classes = observed_classes - set(expected_classes)
    if missing_classes or unexpected_classes:
        raise ValueError(
            "Subtype donor split requires the complete canonical class set; "
            f"missing={sorted(missing_classes)}, "
            f"unexpected={sorted(unexpected_classes)}"
        )

    positions: dict[str, dict[str, np.ndarray]] = {}
    group_frames = []
    broad_stats = {}
    for broad, expected_subtypes in SUBTYPES_BY_BROAD.items():
        broad_eligible = broad_values.eq(broad) & subtype_values.notna()
        broad_obs = obs.loc[broad_eligible].copy()
        broad_obs["_class"] = broad_obs["Celltype_subtraining"].astype(str)
        observed_groups = (
            broad_obs.groupby("Group_ID", observed=True)
            .agg(
                Data_source=("Data_source", "first"),
                n_cells=("Group_ID", "size"),
                n_subtypes=("_class", "nunique"),
            )
            .reset_index()
        )
        observed_groups["Group_ID"] = observed_groups["Group_ID"].astype(str)
        observed_groups["Data_source"] = observed_groups["Data_source"].astype(str)

        configured = fixed_assignment.loc[
            fixed_assignment["broad_type"].eq(broad)
        ].copy()
        observed_ids = set(observed_groups["Group_ID"])
        configured_ids = set(configured["Group_ID"])
        unexpected_observed = sorted(observed_ids - configured_ids)
        missing_configured = sorted(configured_ids - observed_ids)
        if unexpected_observed or missing_configured:
            raise ValueError(
                f"{broad}: observed subtype donors do not exactly match the "
                "fixed JSON assignment; "
                f"unexpected_observed={unexpected_observed}, "
                f"missing_configured={missing_configured}"
            )

        group_frame = configured.merge(
            observed_groups,
            on="Group_ID",
            how="left",
            validate="one_to_one",
            suffixes=("_configured", "_observed"),
        )
        source_mismatch = group_frame["Data_source_configured"].ne(
            group_frame["Data_source_observed"]
        )
        if source_mismatch.any():
            examples = group_frame.loc[
                source_mismatch,
                ["Group_ID", "Data_source_configured", "Data_source_observed"],
            ].head(5)
            raise ValueError(
                f"{broad}: Data_source in the fixed JSON conflicts with the "
                "observed donor metadata; "
                f"examples={examples.to_dict(orient='records')}"
            )
        group_frame = group_frame.rename(
            columns={"Data_source_configured": "Data_source"}
        ).drop(columns="Data_source_observed")
        group_frame = group_frame[
            [
                "broad_type",
                "Group_ID",
                "Data_source",
                "split",
                "n_cells",
                "n_subtypes",
            ]
        ].sort_values("Group_ID", ignore_index=True)
        split_by_group = dict(zip(group_frame["Group_ID"], group_frame["split"]))
        cell_split = obs["Group_ID"].astype(str).map(split_by_group)
        group_frames.append(group_frame)

        positions[broad] = {}
        for split in ("train", "test"):
            mask = broad_eligible.to_numpy() & cell_split.eq(split).to_numpy()
            positions[broad][split] = np.flatnonzero(mask)
            observed = set(subtype_values.loc[mask].astype(str))
            if observed != set(expected_subtypes):
                raise RuntimeError(
                    f"{broad} / {split}: donor split class coverage mismatch; "
                    f"observed={sorted(observed)}, "
                    f"expected={sorted(expected_subtypes)}"
                )
            donor_counts = (
                obs.iloc[positions[broad][split]]
                .groupby("Celltype_subtraining", observed=True)["Group_ID"]
                .nunique()
            )
            insufficient = {
                subtype: int(donor_counts.get(subtype, 0))
                for subtype in expected_subtypes
                if int(donor_counts.get(subtype, 0))
                < MIN_DONORS_PER_SIDE_PER_SUBTYPE
            }
            if insufficient:
                raise ValueError(
                    f"{broad} / {split}: fixed assignment must contain at "
                    f"least {MIN_DONORS_PER_SIDE_PER_SUBTYPE} donor groups "
                    f"per subtype; observed={insufficient}"
                )
        train_groups = set(
            obs.iloc[positions[broad]["train"]]["Group_ID"].astype(str)
        )
        test_groups = set(
            obs.iloc[positions[broad]["test"]]["Group_ID"].astype(str)
        )
        if train_groups & test_groups:
            raise RuntimeError(f"{broad}: donor groups leak across train/test")

        source_split_coverage = group_frame.groupby(
            "Data_source", observed=True
        )["split"].agg(lambda values: set(values))
        missing_source_sides = {
            source: sorted({"train", "test"} - splits)
            for source, splits in source_split_coverage.items()
            if splits != {"train", "test"}
        }
        if missing_source_sides:
            raise ValueError(
                f"{broad}: every contributing Data_source must occur in both "
                f"splits; missing={missing_source_sides}"
            )

        broad_stats[broad] = {
            "n_groups": int(len(group_frame)),
            "n_train_groups": int(group_frame["split"].eq("train").sum()),
            "n_test_groups": int(group_frame["split"].eq("test").sum()),
            "n_eligible_cells": int(broad_eligible.sum()),
        }

    group_frame = pd.concat(group_frames, axis=0, ignore_index=True)
    return {
        "positions": positions,
        "groups": group_frame,
        "broad_stats": broad_stats,
        "n_broad_donor_assignments": int(len(group_frame)),
        "n_train_assignments": int(group_frame["split"].eq("train").sum()),
        "n_test_assignments": int(group_frame["split"].eq("test").sum()),
        "n_eligible_cells": int(eligible_mask.sum()),
    }


def _materialize_rows(adata, positions: np.ndarray):
    view = adata[positions, :]
    if getattr(adata, "isbacked", False):
        return view.to_memory()
    return view.copy()


def write_subtype_splits(
    adata,
    output_dir: Path,
    *,
    assignment_path: Path = SUBTYPE_ASSIGNMENT_PATH,
) -> pd.DataFrame:
    """Write the fixed donor-held-out splits shared within each broad task."""
    if not adata.obs_names.is_unique:
        raise ValueError("Subtype split requires globally unique cell IDs")

    plan = plan_donor_subtype_split(
        adata.obs,
        assignment_path=assignment_path,
    )
    assignment_file = Path(assignment_path).name
    assignments = plan["positions"]
    paths = subtype_output_paths(output_dir)
    rows = []

    for broad in SUBTYPES_BY_BROAD:
        stats = plan["broad_stats"][broad]
        train_ids = set(adata.obs_names[assignments[broad]["train"]].astype(str))
        test_ids = set(adata.obs_names[assignments[broad]["test"]].astype(str))
        if train_ids & test_ids:
            raise RuntimeError(f"{broad}: train/test cell IDs overlap")

        for split in ("train", "test"):
            positions = assignments[broad][split]
            subset = _materialize_rows(adata, positions)
            split_groups = set(subset.obs["Group_ID"].astype(str))
            subset.uns["subtype_split"] = {
                "broad_type": broad,
                "split": split,
                "strategy": SUBTYPE_SPLIT_STRATEGY,
                "algorithm_version": SUBTYPE_SPLIT_ALGORITHM_VERSION,
                "scope": "within_broad_across_subtypes",
                "assignment_schema_version": SUBTYPE_ASSIGNMENT_SCHEMA_VERSION,
                "assignment_file": assignment_file,
                "broad_column": "Celltype_training",
                "subtype_column": "Celltype_subtraining",
                "sample_column": "Sample_ID",
                "donor_column": "Donor_ID",
                "group_column": "Group_ID",
                "n_groups_broad": stats["n_groups"],
                "n_train_groups_broad": stats["n_train_groups"],
                "n_test_groups_broad": stats["n_test_groups"],
                "n_eligible_cells_broad": stats["n_eligible_cells"],
                "n_groups_in_file": int(len(split_groups)),
            }
            path = paths[(broad, split)]
            atomic_write_h5ad(subset, path)
            rows.append(
                {
                    "broad_type": broad,
                    "split": split,
                    "n_cells": int(subset.n_obs),
                    "n_groups": int(len(split_groups)),
                    "n_genes": int(subset.n_vars),
                    "output": str(path),
                }
            )
            del subset
            gc.collect()

    summary = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    _atomic_write_csv(
        plan["groups"],
        output_dir / "subtype_donor_assignments.csv",
    )
    _atomic_write_csv(summary, output_dir / "subtype_split_summary.csv")
    manifest = {
        "strategy": SUBTYPE_SPLIT_STRATEGY,
        "algorithm_version": SUBTYPE_SPLIT_ALGORITHM_VERSION,
        "scope": "within_broad_across_subtypes",
        "assignment_schema_version": SUBTYPE_ASSIGNMENT_SCHEMA_VERSION,
        "assignment_file": assignment_file,
        "n_broad_donor_assignments": plan["n_broad_donor_assignments"],
        "n_train_assignments": plan["n_train_assignments"],
        "n_test_assignments": plan["n_test_assignments"],
        "n_eligible_cells": plan["n_eligible_cells"],
        "broad_stats": plan["broad_stats"],
        "files": [str(paths[key].name) for key in paths],
    }
    _atomic_write_json(manifest, output_dir / "subtype_split_manifest.json")
    print("[Subtype] fixed per-broad donor-held-out splits complete", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary
