"""Dataset-specific broad-cell-type evaluation policy.

This module is evaluation-only.  It never changes training labels and never
uses expression data.  A relaxed match is allowed only when a source ground
truth is coarser than the fixed broad classes predicted by the models.
Keep this file byte-identical to
``celltype_annotation/utils/broad_evaluation_policy.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


POLICY_VERSION = "source-broad-v5-compatible-metrics"

CANONICAL_BROAD_LABELS = frozenset(
    {
        "B cells",
        "CD4+T cells",
        "CD8+T cells",
        "DC",
        "Endothelial cells",
        "Epithelial cells",
        "Fibroblasts",
        "Macrophages",
        "Mast cells",
        "Monocytes",
        "NK cells",
        "Plasma cells",
        "pDC",
    }
)

CAF_TARGET_LABELS = frozenset(
    {
        "CAF_EndoMT",
        "CAF_adi",
        "CAF_api",
        "CAF_myo",
        "CAF_PN",
        "CAF_infla",
        "Fibroblasts",
    }
)

# Standalone subtype benchmarks use exact label equality.  Keeping these
# source-backed ontologies explicit prevents the broad-label alias rules (most
# importantly cDC -> DC) from collapsing a subtype task.
SUBTYPE_DATASET_EXACT_LABELS = {
    "B_cell_test": frozenset(
        {
            "Follicular B cells",
            "Germinal B cells",
            "MALT B cells",
            "Memory B cells",
            "Naive B cells",
        }
    ),
    "CD4_T_cell_test": frozenset(
        {
            "CD4+Tfh/Th cells",
            "CD4+exhausted T cells",
            "CD4+memory T cells",
            "CD4+naive T cells",
            "CD4+reg T cells",
        }
    ),
    "CD8_T_cell_test": frozenset(
        {
            "CD8+MAIT T cells",
            "CD8+Naive T cells",
            "CD8+exhausted T cells",
            "CD8+memory T cells",
        }
    ),
    "DC_test": frozenset({"DC", "cDC"}),
    "Plasma_cell_test": frozenset(
        {"IgA+ Plasma", "IgG+ Plasma", "Plasma cells", "Plasmablasts"}
    ),
}

IMMUNE_CHILDREN = frozenset(
    {
        "B cells",
        "CD4+T cells",
        "CD8+T cells",
        "DC",
        "Macrophages",
        "Mast cells",
        "Monocytes",
        "NK cells",
        "Plasma cells",
        "pDC",
    }
)


def _rule(accepted, level, reason):
    return {
        "accepted": frozenset(accepted),
        "level": level,
        "reason": reason,
    }


# Rules are deliberately separated by test dataset.  No tolerance silently
# propagates to another dataset with a different source annotation system.
DATASET_RULES = {
    # Jerby-Arnon melanoma: the author label T.cell has no CD4/CD8 split.
    "GSE115978": {
        "T.cells": _rule(
            {"CD4+T cells", "CD8+T cells"},
            "parent",
            "source T.cell does not distinguish CD4 from CD8",
        ),
    },
    # Li melanoma: shared-naive T cells lack a reliable CD4/CD8 source label.
    "GSE123139": {
        "T.cells": _rule(
            {"CD4+T cells", "CD8+T cells"},
            "parent",
            "source shared-naive T state does not distinguish CD4 from CD8",
        ),
    },
    # Hanley NSCLC: T.cells and Mono/Mac are explicit coarse author labels;
    # some rows contain only the published Immune/Mesenchymal lineage.
    "GSE153935": {
        "T.cells": _rule(
            {"CD4+T cells", "CD8+T cells"},
            "parent",
            "source T.cells does not distinguish CD4 from CD8",
        ),
        "Mono/Mac": _rule(
            {"Monocytes", "Macrophages"},
            "parent",
            "source Mono/Mac does not distinguish monocytes from macrophages",
        ),
        "Immune": _rule(
            IMMUNE_CHILDREN,
            "lineage",
            "only the source Immune lineage is available",
        ),
        "Mesenchymal": _rule(
            {"Fibroblasts", "Endothelial cells"},
            "lineage",
            "source Mesenchymal lineage contains these study broad classes",
        ),
    },
    # Uhlitz CRC has source-backed broad labels; no broad relaxation is used.
    "GSE166555": {},
    # Zheng CD34+ is outside the trained mature-cell ontology and is excluded,
    # not relaxed into a guessed mature lineage.
    "Zheng_sorted": {},
    # Habermann/scDiffusion reports one undivided T-cell parent class.
    "lung_full": {
        "T.cells": _rule(
            {"CD4+T cells", "CD8+T cells"},
            "parent",
            "source T Cells does not distinguish CD4 from CD8",
        ),
    },
    "sapiens_full": {},
    "CAF_test": {},
}

DATASET_ALIASES = {
    "zhengsorted": "Zheng_sorted",
    "zheng_sorted": "Zheng_sorted",
    "caf_test": "CAF_test",
}

DATASET_EXACT_LABELS = {
    "CAF_test": CAF_TARGET_LABELS,
    **SUBTYPE_DATASET_EXACT_LABELS,
}

LABEL_ALIASES = {
    "b cell": "B cells",
    "b cells": "B cells",
    "b.cell": "B cells",
    "b.cells": "B cells",
    "cd4 t cells": "CD4+T cells",
    "cd4+ t cells": "CD4+T cells",
    "cd4+t cells": "CD4+T cells",
    "cd8 t cells": "CD8+T cells",
    "cd8+ t cells": "CD8+T cells",
    "cd8+t cells": "CD8+T cells",
    "cdc": "DC",
    "dendritic": "DC",
    "dendritic cells": "DC",
    "endothelial": "Endothelial cells",
    "endothelial cells": "Endothelial cells",
    "epithelial": "Epithelial cells",
    "epithelial cells": "Epithelial cells",
    "fibroblast": "Fibroblasts",
    "fibroblasts": "Fibroblasts",
    "immune": "Immune",
    "macrophage": "Macrophages",
    "macrophages": "Macrophages",
    "mast": "Mast cells",
    "mast cells": "Mast cells",
    "mesenchymal": "Mesenchymal",
    "mono/mac": "Mono/Mac",
    "monocytes/macrophages": "Mono/Mac",
    "monocyte": "Monocytes",
    "monocytes": "Monocytes",
    "nk": "NK cells",
    "nk cells": "NK cells",
    "pdc": "pDC",
    "plasma": "Plasma cells",
    "plasma cells": "Plasma cells",
    "proliferating t cells": "T.cells",
    "t cell": "T.cells",
    "t cells": "T.cells",
    "t.cell": "T.cells",
    "t.cells": "T.cells",
}


def normalize_dataset_name(dataset) -> str:
    """Return the policy key for a path, filename, or dataset label."""
    name = Path(str(dataset)).stem
    return DATASET_ALIASES.get(name.lower(), name)


def normalize_broad_label(value):
    """Canonicalize formatting aliases without adding biological tolerance."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "<na>"}:
        return None
    if text in CANONICAL_BROAD_LABELS or text in {
        "T.cells", "Mono/Mac", "Immune", "Mesenchymal"
    }:
        return text
    return LABEL_ALIASES.get(text.lower(), text)


def dataset_rules(dataset):
    return DATASET_RULES.get(normalize_dataset_name(dataset), {})


def exact_labels(dataset):
    """Return the exact target ontology for one evaluation dataset."""
    name = normalize_dataset_name(dataset)
    return DATASET_EXACT_LABELS.get(name, CANONICAL_BROAD_LABELS)


def normalize_evaluation_label(dataset, value):
    """Preserve exact task labels before applying broad-label aliases."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "<na>"}:
        return None
    if text in exact_labels(dataset):
        return text
    return normalize_broad_label(value)


def accepted_predictions(dataset, ground_truth):
    """Return accepted canonical predictions for one source ground truth."""
    truth = normalize_evaluation_label(dataset, ground_truth)
    if truth is None:
        return frozenset()
    rule = dataset_rules(dataset).get(truth)
    if rule is not None:
        return rule["accepted"]
    return frozenset({truth}) if truth in exact_labels(dataset) else frozenset()


def is_exact_ground_truth(dataset, ground_truth) -> bool:
    truth = normalize_evaluation_label(dataset, ground_truth)
    return truth in exact_labels(dataset)


def is_evaluable_ground_truth(dataset, ground_truth) -> bool:
    return bool(accepted_predictions(dataset, ground_truth))


def compare_predictions(y_true, y_pred, dataset):
    """Return normalized strict/relaxed decisions and projected predictions.

    ``relaxed_prediction`` projects an accepted child prediction to a coarse
    source parent. The evaluator uses this as its primary prediction for all
    classification metrics and confusion matrices; ``strict_correct`` remains
    internal exact-match audit information.
    """
    truths = np.asarray(
        [normalize_evaluation_label(dataset, value) for value in y_true],
        dtype=object,
    )
    predictions = np.asarray(
        [
            normalize_evaluation_label(dataset, value) or "Unknown"
            for value in y_pred
        ],
        dtype=object,
    )
    strict = truths == predictions
    relaxed = np.zeros(len(truths), dtype=bool)
    projected = predictions.copy()
    rule_names = np.full(len(truths), "exact", dtype=object)
    levels = np.full(len(truths), "exact", dtype=object)
    rules = dataset_rules(dataset)

    for index, (truth, prediction) in enumerate(zip(truths, predictions)):
        accepted = accepted_predictions(dataset, truth)
        relaxed[index] = strict[index] or prediction in accepted
        rule = rules.get(truth)
        if rule is not None:
            rule_names[index] = (
                f"{truth} -> {'|'.join(sorted(accepted))}"
            )
            levels[index] = rule["level"]
        if relaxed[index] and prediction != truth:
            projected[index] = truth

    return {
        "truth": truths,
        "prediction": predictions,
        "strict_correct": strict,
        "relaxed_correct": relaxed,
        "relaxed_prediction": projected,
        "relaxation_rule": rule_names,
        "relaxation_level": levels,
    }
