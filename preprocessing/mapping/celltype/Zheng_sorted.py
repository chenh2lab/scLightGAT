from mapping.common import broad_from_label, normalize_label

MAJOR_COLUMNS = ["benchmark_label"]

EXPLICIT = {
    "cd19b": "B cells",
    "cd4cd25treg": "CD4+T cells",
    "cd4cd45romemory": "CD4+T cells",
    "cd4cd45racd25naivet": "CD4+T cells",
    "cd4thelper2": "CD4+T cells",
    "cd8cd45ranaivecytotoxic": "CD8+T cells",
    "cd8cytotoxict": "CD8+T cells",
    "cd14monocyte": "Monocytes",
    "cd16monocyte": "Monocytes",
    "cd56nk": "NK cells",
    "dendritic": "DC",
}


def map_celltype(value):
    label = normalize_label(value)
    if label in EXPLICIT:
        return EXPLICIT[label]
    return broad_from_label(value)
