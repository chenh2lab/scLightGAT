from mapping.common import (
    broad_from_label,
    has_cd_marker,
    broad_from_harmonized_source,
    normalize_label,
)
from mapping.source.GSE131907 import MAPPING

PROVENANCE = (
    "original:GEO Cell_subtype; fallback original Cell_type.refined/Cell_type"
)


def _usable(value):
    return value is not None and normalize_label(value) not in {
        "", "na", "nan", "none", "unknown", "undetermined", "unassigned"
    }


def resolve_celltype_row(row):
    # Prefer the finest published annotation whenever it deterministically maps
    # to the scLightGAT-compatible vocabulary.
    for column in ("Cell_subtype", "cell_subtype"):
        if column not in row or not _usable(row[column]):
            continue
        fine = normalize_label(row[column])
        if "smoothmuscle" in fine or fine in {"pericyte", "pericytes"}:
            return None, column, str(row[column])
        if "cd207" in fine or "langerhans" in fine or fine.endswith("lcs"):
            return "DC", column, str(row[column])
        if has_cd_marker(row[column], "cd8") and not has_cd_marker(row[column], "cd4"):
            return "CD8+T cells", column, str(row[column])
        harmonized = MAPPING.get(fine)
        if harmonized is not None:
            return broad_from_harmonized_source(harmonized), column, str(row[column])

    # Some published fine labels (e.g. malignant/undetermined states) are not
    # part of the subtype table.  Fall back to the source broad/refined label
    # instead of inventing a subtype or returning a missing major label.
    for column in (
        "Cell_type.refined",
        "Cell_type_refined",
        "cell_type_refined",
        "Cell_type",
        "cell_type",
    ):
        if column not in row or not _usable(row[column]):
            continue
        mapped = broad_from_label(row[column])
        if _usable(mapped):
            return mapped, column, str(row[column])
    return None, None, None


def map_celltype_row(row):
    return resolve_celltype_row(row)[0]


def map_coarse_celltype_row(row):
    fine = normalize_label(row.get("Cell_subtype", ""))
    if "smoothmuscle" in fine:
        return "Smooth Muscle Cells"
    if fine in {"pericyte", "pericytes"}:
        return "Pericytes"
    if has_cd_marker(row.get("Cell_subtype", ""), "cd4") and has_cd_marker(
        row.get("Cell_subtype", ""), "cd8"
    ):
        return "T.cells"
    for column in ("Cell_type.refined", "Cell_type_refined", "Cell_type"):
        if column not in row:
            continue
        value = row.get(column)
        label = normalize_label(value)
        if label in {"t", "tcell", "tcells", "tlymphocyte", "tlymphocytes"}:
            return "T.cells"
        if label in {"tnkcells", "tandnkcells"}:
            return "T/NK cells"
        if label in {"myeloid", "myeloidcells"}:
            return "Mono/Mac"
        if label:
            return str(value)
    return "Unknown"
