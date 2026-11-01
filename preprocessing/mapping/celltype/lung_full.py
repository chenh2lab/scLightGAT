from __future__ import annotations

MAJOR_COLUMNS = ["source_celltype"]
EVALUATION_COARSE_LABELS = {"T.cells"}

# Explicit 31-class Habermann/scDiffusion source-label harmonization.
# The source labels are preserved in obs["source_celltype"].
MAPPING = {
    "AT1": "Epithelial cells",
    "AT2": "Epithelial cells",
    "Basal": "Epithelial cells",
    "Ciliated": "Epithelial cells",
    "Differentiating Ciliated": "Epithelial cells",
    "KRT5-/KRT17+": "Epithelial cells",
    "MUC5AC+ High": "Epithelial cells",
    "MUC5B+": "Epithelial cells",
    "SCGB3A2+": "Epithelial cells",
    "SCGB3A2+ SCGB1A1+": "Epithelial cells",
    "Proliferating Epithelial Cells": "Epithelial cells",
    "Transitional AT2": "Epithelial cells",
    "Mesothelial Cells": "Epithelial cells",
    "Endothelial Cells": "Endothelial cells",
    "Lymphatic Endothelial Cells": "Endothelial cells",
    "Fibroblasts": "Fibroblasts",
    "HAS1 High Fibroblasts": "Fibroblasts",
    "Myofibroblasts": "Fibroblasts",
    "PLIN2+ Fibroblasts": "Fibroblasts",
    "Macrophages": "Macrophages",
    "Proliferating Macrophages": "Macrophages",
    "Monocytes": "Monocytes",
    "Mast Cells": "Mast cells",
    "B Cells": "B cells",
    "Plasma Cells": "Plasma cells",
    "NK Cells": "NK cells",
    "cDCs": "DC",
    "pDCs": "pDC",
}


UNRESOLVED_SOURCE_LABELS = {
    "Smooth Muscle Cells",
    "T Cells",
    "Proliferating T Cells",
}


def map_celltype(value):
    key = str(value).strip()
    if key in UNRESOLVED_SOURCE_LABELS:
        return None
    if key not in MAPPING:
        raise ValueError(f"lung_full: unmapped source cell type: {value!r}")
    return MAPPING[key]


def map_coarse_celltype_row(row):
    value = str(row.get("source_celltype", "")).strip()
    if value in {"T Cells", "Proliferating T Cells"}:
        return "T.cells"
    if value == "Smooth Muscle Cells":
        return "Smooth Muscle Cells"
    return "Unknown"
