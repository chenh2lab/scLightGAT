from mapping.common import broad_from_label, normalize_label

# Resolve each cell inside its published epithelial/immune/stromal compartment.
MAJOR_COLUMNS = ["main_cell_type"]


def resolve_celltype_row(row):
    compartment = normalize_label(row.get("main_cell_type", ""))
    if compartment == "epithelial":
        return "Epithelial cells", "main_cell_type", str(row.get("main_cell_type"))
    if compartment == "stromal":
        label = normalize_label(row.get("cell_type_str_simple", ""))
        if label == "endothelial":
            return "Endothelial cells", "cell_type_str_simple", str(row.get("cell_type_str_simple"))
        if label in {"cafs", "cbfbs", "myofbs", "ucfbs", "cclfbs"}:
            return "Fibroblasts", "cell_type_str_simple", str(row.get("cell_type_str_simple"))
        return None, "cell_type_str_simple", str(row.get("cell_type_str_simple"))
    if compartment == "immune":
        label = normalize_label(row.get("cell_type_imm_simple", ""))
        if label == "tconv":
            return "CD4+T cells", "cell_type_imm_simple", str(row.get("cell_type_imm_simple"))
        if label == "treg":
            return "CD4+T cells", "cell_type_imm_simple", str(row.get("cell_type_imm_simple"))
        if label.startswith("plasma"):
            return "Plasma cells", "cell_type_imm_simple", str(row.get("cell_type_imm_simple"))
        raw = row.get("cell_type_imm_simple", "")
        return broad_from_label(raw), "cell_type_imm_simple", str(raw)
    return None, "main_cell_type", str(row.get("main_cell_type"))


def map_celltype_row(row):
    return resolve_celltype_row(row)[0]


def map_coarse_celltype_row(row):
    compartment = normalize_label(row.get("main_cell_type", ""))
    if compartment == "stromal":
        value = str(row.get("cell_type_str_simple", ""))
        if normalize_label(value) in {"pericyte", "pericytes", "glia"}:
            return value
    return "Unknown"


PROVENANCE = "original main_cell_type with compartment-specific simple annotation"
