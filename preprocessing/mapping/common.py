from __future__ import annotations

import re


def normalize_label(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def has_cd_marker(value, marker: str) -> bool:
    """Recognize CD4/CD8 lineage tokens without reading CD45/CD48 as CD4."""
    marker = marker.lower()
    if marker not in {"cd4", "cd8"}:
        raise ValueError(f"Unsupported CD marker: {marker}")
    raw = str(value).lower()
    label = normalize_label(value)
    if re.search(rf"(?<![a-z0-9]){marker}(?!\d)", raw):
        return True
    if label in {"cd4cd8", "cd8cd4"} or label.startswith(("cd4cd8", "cd8cd4")):
        return True
    if label.endswith(marker):
        return True
    return label.startswith(marker) and (
        len(label) == len(marker) or not label[len(marker)].isdigit()
    )


B_SOURCE_LABELS = {
    "Follicular B cells",
    "Germinal B cells",
    "MALT B cells",
    "Memory B cells",
    "Naive B cells",
}
CD4_SOURCE_LABELS = {
    "CD4+Tfh/Th cells",
    "CD4+exhausted T cells",
    "CD4+memory T cells",
    "CD4+naive T cells",
    "CD4+reg T cells",
}
CD8_SOURCE_LABELS = {
    "CD8+MAIT T cells",
    "CD8+Naive T cells",
    "CD8+exhausted T cells",
    "CD8+memory T cells",
}
DC_SOURCE_LABELS = {"DC", "cDC"}
PLASMA_SOURCE_LABELS = {
    "IgA+ Plasma",
    "IgG+ Plasma",
    "Plasma cells",
    "Plasmablasts",
}


def broad_from_harmonized_source(value):
    if value in B_SOURCE_LABELS:
        return "B cells"
    if value in CD4_SOURCE_LABELS:
        return "CD4+T cells"
    if value in CD8_SOURCE_LABELS:
        return "CD8+T cells"
    if value in DC_SOURCE_LABELS:
        return "DC"
    if value in PLASMA_SOURCE_LABELS:
        return "Plasma cells"
    # Source fine labels that are safely collapsible to the thesis fibroblast
    # class. Smooth-muscle/perivascular labels are deliberately NOT included.
    if value in {"CAFs", "Fibroblasts", "Myofibroblasts"}:
        return "Fibroblasts"
    return value


def broad_from_label(value):
    """Conservative source-label -> thesis major harmonization.

    Ambiguous mixed/coarse labels remain unresolved.  Rules intentionally avoid
    substring traps such as T/NK -> NK, CD4/CD8 -> CD4, STAT1 -> AT1, or LAMP3
    -> LAM macrophage.
    """
    label = normalize_label(value)
    if label in {"", "na", "nan", "none", "unknown", "undetermined", "unassigned"}:
        return None

    # Explicit mixed lineages are not deterministically one thesis class.
    has_cd4 = has_cd_marker(value, "cd4")
    has_cd8 = has_cd_marker(value, "cd8")
    if has_cd4 and has_cd8:
        return None
    if label.startswith("tnk") or label in {"tandnkcells", "tornkcells", "tnkcell"}:
        return None

    exact = {
        "b": "B cells",
        "endo": "Endothelial cells",
        "dc": "DC",
        "dcs": "DC",
        "pdc": "pDC",
        "pdcs": "pDC",
        "follicular": "B cells",   # GSE166555 B state
        "gc": "B cells",           # GSE166555 B state
        "germinal": "B cells",
        "treg": "CD4+T cells",
        "tregs": "CD4+T cells",
        "cd4memory": "CD4+T cells",
        "plasma": "Plasma cells",
        "plasmacell": "Plasma cells",
        "plasmacells": "Plasma cells",
        "nk": "NK cells",
        "nks": "NK cells",
        "nkcell": "NK cells",
        "nkcells": "NK cells",
        "naturalkiller": "NK cells",
        "naturalkillercells": "NK cells",
        "at1": "Epithelial cells",
        "at2": "Epithelial cells",
    }
    if label in exact:
        return exact[label]

    # CD4/CD8 semantics precede B-cell words such as follicular.
    if has_cd4 or any(x in label for x in (
        "tfollicularhelper", "follicularhelpert", "tfh",
        "regulatoryt", "treg", "thelper", "th17",
    )):
        return "CD4+T cells"
    if has_cd8 or "mait" in label:
        return "CD8+T cells"

    # Plasma before generic B lineage.
    if any(x in label for x in (
        "plasmablast", "plasmacell", "igaplasma", "iggplasma",
    )):
        return "Plasma cells"

    if any(x in label for x in (
        "bcell", "blymphocyte", "follicularb", "germinalb",
        "memoryb", "naiveb", "maltb",
    )):
        return "B cells"

    if "plasmacytoiddc" in label or label.startswith("pdc"):
        return "pDC"
    if any(x in label for x in (
        "dendritic", "cdc", "dc1", "dc2", "activateddc", "lamp3dc",
    )):
        return "DC"

    if label.startswith("nk") or "naturalkiller" in label:
        return "NK cells"
    if "macroph" in label or re.match(r"^lam[12](?:$|[a-z])", label):
        return "Macrophages"
    if "monocyt" in label:
        return "Monocytes"
    if "mast" in label:
        return "Mast cells"

    if "endothelial" in label or "lymphaticendothelial" in label:
        return "Endothelial cells"
    if any(x in label for x in (
        "fibroblast", "myofibroblast", "matrixfb", "caf",
    )):
        return "Fibroblasts"

    if any(x in label for x in (
        "epithelial", "hepatocyte", "cholangiocyte", "alveolartype1",
        "alveolartype2", "ciliated", "clubcell", "basalepithelial", "luminal",
    )):
        return "Epithelial cells"

    return None

