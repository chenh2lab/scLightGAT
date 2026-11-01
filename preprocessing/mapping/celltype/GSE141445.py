from mapping.common import broad_from_label, normalize_label

# Same manuscript-era scATOMIC preserved-reference schema used for GSE132465.
# Prediction columns (layer_* / scATOMIC_pred) are never used as ground truth.
MAJOR_COLUMNS = [
    "harmonized_original_annotation",
    "original_annotation.1",
    "Cell_type",
]

def map_celltype(value):
    label = normalize_label(value)
    if label in {"prostatecancercell", "basalintermediate"}:
        return "Epithelial cells"
    if label in {"t", "tcell", "tcells", "macrophagemonocytemyeloid"}:
        return None
    return broad_from_label(value)


def map_coarse_celltype_row(row):
    for column in MAJOR_COLUMNS:
        label = normalize_label(row.get(column, ""))
        if label in {"t", "tcell", "tcells"}:
            return "T.cells"
        if label == "macrophagemonocytemyeloid":
            return "Mono/Mac"
    return "Unknown"
PROVENANCE = "scATOMIC preserved source/reference metadata (harmonized_original_annotation; source-backed fallbacks only)"
