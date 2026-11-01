from mapping.common import broad_from_label, normalize_label

PROVENANCE = "Li2018:all_mc_group/t_nk_mc_group/non_t_nk_mc_group"

CD4_STATES = {"treg", "tfh", "emcd4", "dysfcd4"}
CD8_STATES = {"dysfunctional", "transitional", "cytotoxic"}
CD4_SOURCE_LABELS = {"cd4", "cd4t", "cd4tcells", "cd4positivealphabettcell"}
CD8_SOURCE_LABELS = {"cd8", "cd8t", "cd8tcells", "cd8positivealphabettcell"}
EVALUATION_COARSE_LABELS = {"T.cells"}


def resolve_celltype_row(row):
    t_state = normalize_label(row.get("t_nk_mc_group", ""))
    if t_state in CD4_STATES:
        return "CD4+T cells", "t_nk_mc_group", str(row.get("t_nk_mc_group"))
    if t_state in CD8_STATES:
        return "CD8+T cells", "t_nk_mc_group", str(row.get("t_nk_mc_group"))
    if t_state == "nk":
        return "NK cells", "t_nk_mc_group", str(row.get("t_nk_mc_group"))
    if t_state == "naive":
        original = normalize_label(row.get("Cell.type", ""))
        raw = f"t_nk_mc_group={row.get('t_nk_mc_group')} | Cell.type={row.get('Cell.type')}"
        # Exact normalized labels only.  Substring matching incorrectly read
        # the source sorting gate ``CD45`` as CD4 and assigned all shared-naive
        # cells to CD4.
        if original in CD4_SOURCE_LABELS:
            return "CD4+T cells", "t_nk_mc_group+Cell.type", raw
        if original in CD8_SOURCE_LABELS:
            return "CD8+T cells", "t_nk_mc_group+Cell.type", raw
        return None, "t_nk_mc_group+Cell.type", raw

    non_t = row.get("non_t_nk_mc_group")
    if non_t is not None and str(non_t).lower() not in {"nan", "none", ""}:
        if normalize_label(non_t) == "maturedc":
            return "DC", "non_t_nk_mc_group", str(non_t)
        return broad_from_label(non_t), "non_t_nk_mc_group", str(non_t)

    all_group = normalize_label(row.get("all_mc_group", ""))
    if all_group == "nk":
        return "NK cells", "all_mc_group", str(row.get("all_mc_group"))
    raw = row.get("all_mc_group", "")
    return broad_from_label(raw), "all_mc_group", str(raw)


def map_celltype_row(row):
    return resolve_celltype_row(row)[0]


def map_coarse_celltype_row(row):
    if normalize_label(row.get("t_nk_mc_group", "")) == "naive":
        return "T.cells"
    non_t = normalize_label(row.get("non_t_nk_mc_group", ""))
    if non_t in {"osteoclast", "osteoclasts"}:
        return "Osteoclast"
    if normalize_label(row.get("all_mc_group", "")) == "tumor":
        return "Tumor"
    return "Unknown"
