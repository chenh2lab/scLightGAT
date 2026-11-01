from mapping.common import broad_from_label, normalize_label

MAJOR_COLUMNS = ["Cell.type2", "Cell.type", "Lineage"]
EVALUATION_COARSE_LABELS = {"T.cells", "Mono/Mac", "Immune", "Mesenchymal"}
map_celltype = broad_from_label


def map_coarse_celltype_row(row):
    for column in MAJOR_COLUMNS:
        value = row.get(column)
        label = normalize_label(value)
        if label in {"t", "tcell", "tcells"}:
            return "T.cells"
        if label in {"monomac", "monocytesmacrophages"}:
            return "Mono/Mac"
        if label == "immune":
            return "Immune"
        if label == "mesenchymal":
            return "Mesenchymal"
    return "Unknown"
