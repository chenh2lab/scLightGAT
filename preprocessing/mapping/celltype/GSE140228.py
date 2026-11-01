from mapping.common import broad_from_label, normalize_label

MAJOR_COLUMNS = ["celltype_sub"]


def map_celltype(value):
    raw = str(value)
    label = normalize_label(value)

    # Mixed CD4/CD8 clusters are not deterministically assignable to one
    # thesis broad class; exclude them rather than inventing a lineage.
    if raw.startswith("CD4/CD8-"):
        return None
    if raw.startswith("CD4-"):
        return "CD4+T cells"
    if raw.startswith("CD8-"):
        return "CD8+T cells"
    if raw == "Lymphoid-B-Plasma":
        return "Plasma cells"
    if raw == "Lymphoid-B":
        return "B cells"
    if raw.startswith("NK-"):
        return "NK cells"
    if label == "ilcs":
        return None
    if raw.startswith("Mφ-"):
        return "Macrophages"
    if raw.startswith("Mono-"):
        return "Monocytes"
    if raw.startswith("DC-"):
        return "DC"
    if raw.startswith("Mast-"):
        return "Mast cells"
    return broad_from_label(value)


def map_coarse_celltype_row(row):
    value = str(row.get("celltype_sub", ""))
    if value.startswith("CD4/CD8-"):
        return "T.cells"
    return "Unknown"
