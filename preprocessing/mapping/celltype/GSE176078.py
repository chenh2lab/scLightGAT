from mapping.common import (
    broad_from_label,
    has_cd_marker,
    broad_from_harmonized_source,
    normalize_label,
)
from mapping.source.GSE176078 import MAPPING

PROVENANCE = (
    "original:celltype_subset/celltype_minor; fallback original:celltype_major"
)


def resolve_celltype_row(row):
    for column in ("celltype_subset", "celltype_minor", "subtype"):
        if column not in row or row[column] is None:
            continue
        text = str(row[column])
        label = normalize_label(text)
        if label in {"", "nan", "none", "na"}:
            continue

        harmonized = MAPPING.get(label)
        if harmonized is not None:
            return broad_from_harmonized_source(harmonized), column, text

        # The Wu et al. T-cell subset names explicitly encode CD4/CD8.
        if has_cd_marker(text, "cd4"):
            return "CD4+T cells", column, text
        if has_cd_marker(text, "cd8"):
            return "CD8+T cells", column, text
        if "nkcells" in label and "nkt" not in label:
            return "NK cells", column, text
        if "pdc" in label:
            return "pDC", column, text
        if "cdc" in label or "dclamp3" in label:
            return "DC", column, text
        if "macroph" in label or "lam" in label:
            return "Macrophages", column, text
        if "monocyte" in label:
            return "Monocytes", column, text
        # PVL/perivascular-like is not one of the thesis 13 broad classes and
        # is not biologically identical to fibroblast.  Leave it unresolved
        # rather than creating an extra class or forcing a dubious collapse.
        if label in {"pvl", "perivascularlike", "perivascularlikecells"}:
            return None, column, text

    if "celltype_major" in row:
        text = str(row["celltype_major"])
        major_label = normalize_label(row["celltype_major"])
        if major_label in {"pvl", "perivascularlike", "perivascularlikecells"}:
            return None, "celltype_major", text
        return broad_from_label(row["celltype_major"]), "celltype_major", text
    return None, None, None


def map_celltype_row(row):
    return resolve_celltype_row(row)[0]


def map_coarse_celltype_row(row):
    for column in ("celltype_subset", "celltype_minor", "subtype", "celltype_major"):
        value = row.get(column)
        label = normalize_label(value)
        if label in {"t", "tcell", "tcells"} or (
            label.startswith("tcells") and "nkt" not in label and "nkcells" not in label
        ):
            return "T.cells"
        if "nkt" in label:
            return "T/NK cells"
        if label == "cyclingmyeloid":
            return "Mono/Mac"
        if label.startswith("pvl") or label == "cyclingpvl":
            return "PVL"
        if label:
            return str(value)
    return "Unknown"
