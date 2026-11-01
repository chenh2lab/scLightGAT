from __future__ import annotations

MAJOR_COLUMNS = ["source_celltype"]

# Explicit harmonization used for the six Tabula Sapiens labels in the
# scDiffusion-derived benchmark. Original labels remain in obs["source_celltype"].
MAPPING = {
    "fibroblast": "Fibroblasts",
    "macrophage": "Macrophages",
    "memory b cell": "B cells",
    "naive b cell": "B cells",
    "nk cell": "NK cells",
    "plasma cell": "Plasma cells",
}


def map_celltype(value):
    key = str(value).strip().lower()
    if key not in MAPPING:
        raise ValueError(f"sapiens_full: unmapped source cell type: {value!r}")
    return MAPPING[key]
