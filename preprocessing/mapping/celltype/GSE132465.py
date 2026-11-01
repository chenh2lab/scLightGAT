from mapping.common import broad_from_label, normalize_label

# scATOMIC preserved-reference metadata as actually exported by the
# manuscript-era training object.  Do not use layer_* / scATOMIC_pred here:
# those are model predictions, not source-backed ground truth.
MAJOR_COLUMNS = [
    "harmonized_original_annotation",
    "original_annotation.1",
    "Cell_type",
]

def map_celltype(value):
    if normalize_label(value) in {"t", "tcell", "tcells", "macrophagemonocytemyeloid"}:
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
