from mapping.common import broad_from_label, normalize_label

MAJOR_COLUMNS = ["cell.types", "cell_type", "CellType"]
EVALUATION_COARSE_LABELS = {"T.cells"}
map_celltype = broad_from_label


def map_coarse_celltype_row(row):
    for column in MAJOR_COLUMNS:
        value = row.get(column)
        label = normalize_label(value)
        if label in {"t", "tcell", "tcells"}:
            return "T.cells"
        if label in {"mal", "malignant"}:
            return "Malignant"
    return "Unknown"
