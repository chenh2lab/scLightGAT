"""
Label Utilities for Cell-type Annotation
========================================
This module provides utilities for cell type label standardization and 
accuracy calculation with custom rules for myeloid grouping and fine-grained tolerance.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Label Mapping Dictionaries
# ============================================================================

# Comprehensive mapping from various label formats to standardized format
LABEL_MAPPING = {
    # T cell variations
    'CD4+ T cells': 'CD4+T cells',
    'CD8+ T cells': 'CD8+T cells',
    'CD4 T cells': 'CD4+T cells',
    'CD8 T cells': 'CD8+T cells',
    'CD4+ Tcells': 'CD4+T cells',
    'CD8+ Tcells': 'CD8+T cells',
    'CD4+Tcell': 'CD4+T cells',
    'CD8+Tcell': 'CD8+T cells',
    'CD4+ T cell': 'CD4+T cells',
    'CD8+ T cell': 'CD8+T cells',
    'T cells': 'CD4+T cells',  # Generic T cells default to CD4+
    'T cell': 'CD4+T cells',
    'T/NK cells': 'NK cells',
    
    # Myeloid variations
    'Monocyets/Macrophages': 'Macrophages',
    'Monocytes/Macrophages': 'Macrophages',
    'MonocytesMacrophages': 'Macrophages',
    'Monocyte': 'Monocytes',
    'Macrophage': 'Macrophages',
    
    # DC variations
    'DCs': 'DC',
    'Dendritic cells': 'DC',
    'Dendritic cell': 'DC',
    'cDC': 'DC',
    'pDCs': 'pDC',
    
    # B cell variations
    'B cell': 'B cells',
    'Bcells': 'B cells',
    'B-cells': 'B cells',
    
    # NK cell variations
    'NK cell': 'NK cells',
    'NKcells': 'NK cells',
    'NK': 'NK cells',
    
    # Plasma cell variations
    'Plasma cell': 'Plasma cells',
    'Plasmacells': 'Plasma cells',
    
    # Fibroblast variations
    'Fibroblast': 'Fibroblasts',
    'CAF': 'Fibroblasts',
    'CAFs': 'Fibroblasts',
    
    # Epithelial variations
    'Epithelial cell': 'Epithelial cells',
    'Epithelium': 'Epithelial cells',
    'Epithelialcells': 'Epithelial cells',
    
    # Endothelial variations
    'Endothelial cell': 'Endothelial cells',
    'Endothelium': 'Endothelial cells',
    'Endothelialcells': 'Endothelial cells',
    
    # Mast cell variations
    'Mast cell': 'Mast cells',
    'Mastcells': 'Mast cells',

    # Common lowercase / dataset-specific variants
    'fibroblast': 'Fibroblasts',
    'fibroblasts': 'Fibroblasts',
    'macrophage': 'Macrophages',
    'macrophages': 'Macrophages',
    'monocyte': 'Monocytes',
    'monocytes': 'Monocytes',
    'memory b cell': 'B cells',
    'memory b cells': 'B cells',
    'naive b cell': 'B cells',
    'naive b cells': 'B cells',
    'plasma cell': 'Plasma cells',
    'plasma cells': 'Plasma cells',
    'nk cell': 'NK cells',
    'nk cells': 'NK cells',
    'epithelial cell': 'Epithelial cells',
    'epithelial cells': 'Epithelial cells',
    'endothelial cell': 'Endothelial cells',
    'endothelial cells': 'Endothelial cells',
}


def _normalize_label_key(text: str) -> str:
    """Normalize label text for robust lookup across formatting variants."""
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("_", " ").replace("-", " "))


LABEL_MAPPING_NORMALIZED = {
    _normalize_label_key(src): dst for src, dst in LABEL_MAPPING.items()
}

# ============================================================================
# Myeloid Group Definition (Rule 1.1)
# ============================================================================

# Cells in this group are considered equivalent for accuracy calculation
MYELOID_GROUP = {'Macrophages', 'Monocytes', 'DC'}
MYELOID_COMBINED_LABEL = 'Macro/Monocytes/DC'

# ============================================================================
# Allowed Cell Types (Rule 2.0) 
# ============================================================================

# Only these cell types are included in accuracy calculation
# All other cell types are skipped/excluded
ALLOWED_CELLTYPES = {
    'Epithelial cells',
    'Macrophages',
    'CD4+T cells',
    'Endothelial cells',
    'Fibroblasts',
    'NK cells',
    'B cells',
    'CD8+T cells',
    'Monocytes',
    'Plasma cells',
    'DC',
    'Mast cells',
    'pDC',
}

# ============================================================================
# Fine-grained to Broad Type Mapping (Rule 1.3)
# ============================================================================

# If a model predicts a fine-grained type but ground truth is broad,
# it should still count as correct
FINE_TO_BROAD = {
    # CD4+ T cell subtypes
    'CD4+Tfh/Th cells': 'CD4+T cells',
    'CD4+exhausted T cells': 'CD4+T cells',
    'CD4+memory T cells': 'CD4+T cells',
    'CD4+naive T cells': 'CD4+T cells',
    'CD4+reg T cells': 'CD4+T cells',
    'CD4+ Tfh/Th cells': 'CD4+T cells',
    'CD4+ exhausted T cells': 'CD4+T cells',
    'CD4+ memory T cells': 'CD4+T cells',
    'CD4+ naive T cells': 'CD4+T cells',
    'CD4+ reg T cells': 'CD4+T cells',
    'Treg': 'CD4+T cells',
    'Tregs': 'CD4+T cells',
    
    # CD8+ T cell subtypes
    'CD8+MAIT T cells': 'CD8+T cells',
    'CD8+Naive T cells': 'CD8+T cells',
    'CD8+exhausted T cells': 'CD8+T cells',
    'CD8+memory T cells': 'CD8+T cells',
    'CD8+ MAIT T cells': 'CD8+T cells',
    'CD8+ Naive T cells': 'CD8+T cells',
    'CD8+ exhausted T cells': 'CD8+T cells',
    'CD8+ memory T cells': 'CD8+T cells',
    
    # B cell subtypes
    'Follicular B cells': 'B cells',
    'Germinal B cells': 'B cells',
    'MALT B cells': 'B cells',
    'Memory B cells': 'B cells',
    'Naive B cells': 'B cells',
    
    # DC subtypes
    'cDC': 'DC',
    'pDC': 'DC',
    
    # Plasma cell subtypes
    'IgA+ Plasma': 'Plasma cells',
    'IgG+ Plasma': 'Plasma cells',
    'Plasmablasts': 'Plasma cells',
    
    # Myeloid subtypes
    'M1 Macrophages': 'Macrophages',
    'M2 Macrophages': 'Macrophages',
    'Classical Monocytes': 'Monocytes',
    'Non-classical Monocytes': 'Monocytes',
}

# ============================================================================
# Standardization Functions
# ============================================================================

def standardize_label(label: str) -> str:
    """
    Standardize a single cell type label.
    
    Args:
        label: Raw cell type label
        
    Returns:
        Standardized label
    """
    if pd.isna(label):
        return label
    
    label = str(label).strip()
    
    # Direct mapping
    if label in LABEL_MAPPING:
        return LABEL_MAPPING[label]

    # Case/format-insensitive mapping
    normalized_key = _normalize_label_key(label)
    if normalized_key in LABEL_MAPPING_NORMALIZED:
        return LABEL_MAPPING_NORMALIZED[normalized_key]
    
    # Try without parentheses content
    if '(' in label:
        base_label = label.split('(')[0].strip()
        if base_label in LABEL_MAPPING:
            return LABEL_MAPPING[base_label]
        base_key = _normalize_label_key(base_label)
        if base_key in LABEL_MAPPING_NORMALIZED:
            return LABEL_MAPPING_NORMALIZED[base_key]
    
    return label


def standardize_labels_series(labels: pd.Series) -> pd.Series:
    """
    Standardize a series of cell type labels.
    
    Args:
        labels: Series of cell type labels
        
    Returns:
        Series with standardized labels
    """
    return labels.apply(standardize_label)


def standardize_adata_labels(adata, column: str = 'Ground Truth', inplace: bool = True):
    """
    Standardize cell type labels in an AnnData object.
    
    Args:
        adata: AnnData object
        column: Column name containing labels
        inplace: Whether to modify in place
        
    Returns:
        Modified AnnData if inplace=False, else None
    """
    if not inplace:
        adata = adata.copy()
    
    if column in adata.obs.columns:
        adata.obs[column] = standardize_labels_series(adata.obs[column].astype(str))
    
    if not inplace:
        return adata


def collapse_major_type_label(label: str) -> str:
    """
    Collapse myeloid-related labels into a single major-type label.
    Only applies to labels in ALLOWED_CELLTYPES.

    Args:
        label: Raw or standardized cell type label

    Returns:
        Collapsed label for major-type confusion matrix usage,
        or None if label is not in ALLOWED_CELLTYPES
    """
    standardized_label = standardize_label(label)

    # Keep pre-merged myeloid label as-is so datasets that already merged
    # Macrophages/Monocytes/DC are still included in major-type evaluation.
    if standardized_label == MYELOID_COMBINED_LABEL:
        return MYELOID_COMBINED_LABEL
    
    # Return None for non-allowed cell types
    if standardized_label not in ALLOWED_CELLTYPES:
        return None
    
    broad_label = get_broad_type(standardized_label)
    if broad_label in MYELOID_GROUP:
        return MYELOID_COMBINED_LABEL
    return broad_label


def collapse_major_type_labels(labels: pd.Series) -> pd.Series:
    """
    Collapse a series of labels for major-type confusion matrix usage.
    Non-allowed cell types are converted to None.
    """
    result = labels.apply(collapse_major_type_label)
    # Return series with None values for non-allowed cell types
    return result


# ============================================================================
# Accuracy Calculation Functions (Rules 1.1, 1.2, 1.3, 1.4)
# ============================================================================

def get_broad_type(label: str) -> str:
    """Get broad type for a fine-grained label."""
    return FINE_TO_BROAD.get(label, label)


def is_myeloid(label: str) -> bool:
    """Check if a label belongs to myeloid group."""
    return label in MYELOID_GROUP or get_broad_type(label) in MYELOID_GROUP


def labels_match(y_true: str, y_pred: str, 
                 myeloid_group: bool = True,
                 fine_grained_tolerance: bool = True,
                 epithelial_tolerance: bool = True) -> bool:
    """
    Check if two labels match according to custom rules.
    
    Args:
        y_true: Ground truth label
        y_pred: Predicted label
        myeloid_group: If True, Macrophages/Monocytes/DC are equivalent
        fine_grained_tolerance: If True, fine-grained predictions matching broad GT are correct
        epithelial_tolerance: If True, Epithelial cells predictions are never wrong
        
    Returns:
        True if labels match according to rules
    """
    # Standardize labels first
    y_true = standardize_label(str(y_true))
    y_pred = standardize_label(str(y_pred))
    
    # Exact match
    if y_true == y_pred:
        return True
    
    # # Rule 1.4: Epithelial cells tolerance
    # if epithelial_tolerance and y_pred == 'Epithelial cells':
    #     return True
    
    # Rule 1.1: Myeloid group equivalence
    if myeloid_group:
        if is_myeloid(y_true) and is_myeloid(y_pred):
            return True
    
    # Rule 1.3: Fine-grained to broad tolerance
    if fine_grained_tolerance:
        # If prediction is more fine-grained than ground truth
        pred_broad = get_broad_type(y_pred)
        if pred_broad == y_true:
            return True
        
        # If ground truth is more fine-grained than prediction
        true_broad = get_broad_type(y_true)
        if true_broad == y_pred:
            return True
        
        # Both could be fine-grained but map to same broad type
        if pred_broad == true_broad and pred_broad != y_pred:
            return True
    
    return False


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series,
                      myeloid_group: bool = True,
                      fine_grained_tolerance: bool = True,
                      epithelial_tolerance: bool = True,
                      allowed_celltypes_only: bool = True,
                      return_details: bool = False) -> Tuple:
    """
    Calculate accuracy with custom rules.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        myeloid_group: If True, Macrophages/Monocytes/DC are equivalent
        fine_grained_tolerance: If True, fine-grained predictions are correct
        epithelial_tolerance: If True, Epithelial cells predictions are never wrong
        allowed_celltypes_only: If True, only include samples where y_true is in ALLOWED_CELLTYPES
        return_details: If True, return detailed breakdown
        
    Returns:
        Tuple of (weighted_accuracy, detailed_report)
    """
    # Standardize labels
    y_true_std = standardize_labels_series(y_true.astype(str))
    y_pred_std = standardize_labels_series(y_pred.astype(str))
    
    # Filter by allowed cell types if requested
    # Only filter by ground truth: predictions outside ALLOWED_CELLTYPES
    # (e.g. CHETAH "Unassigned" / "Node*") should count as incorrect,
    # not be silently excluded.
    if allowed_celltypes_only:
        mask = y_true_std.isin(ALLOWED_CELLTYPES)
        y_true_std = y_true_std[mask]
        y_pred_std = y_pred_std[mask]
    
    # Calculate matches
    matches = []
    match_reasons = []
    
    for i, (gt, pred) in enumerate(zip(y_true_std, y_pred_std)):
        if pd.isna(gt) or pd.isna(pred):
            matches.append(False)
            match_reasons.append('NA')
            continue
        
        is_match = labels_match(
            gt, pred, 
            myeloid_group=myeloid_group,
            fine_grained_tolerance=fine_grained_tolerance,
            epithelial_tolerance=epithelial_tolerance
        )
        matches.append(is_match)
        
        # Determine reason
        if gt == pred:
            match_reasons.append('exact')
        elif is_match:
            if epithelial_tolerance and pred == 'Epithelial cells':
                match_reasons.append('epithelial_tolerance')
            elif myeloid_group and is_myeloid(gt) and is_myeloid(pred):
                match_reasons.append('myeloid_group')
            elif fine_grained_tolerance:
                match_reasons.append('fine_grained_tolerance')
            else:
                match_reasons.append('other')
        else:
            match_reasons.append('mismatch')
    
    matches_array = np.asarray(matches, dtype=float)
    raw_accuracy = float(matches_array.mean()) if len(matches) > 0 else 0.0

    # Weight each ground-truth class equally by assigning inverse-frequency
    # sample weights. This keeps rare cell types from being dominated by the
    # largest classes in the overall accuracy.
    class_support = y_true_std.value_counts().to_dict()
    sample_weights = np.asarray(
        [1.0 / class_support[label] for label in y_true_std],
        dtype=float
    ) if len(matches) > 0 else np.asarray([], dtype=float)
    total_weight = float(sample_weights.sum())
    accuracy = (
        float(np.average(matches_array, weights=sample_weights))
        if total_weight > 0
        else 0.0
    )

    class_breakdown = []
    for label in sorted(class_support):
        class_mask = (y_true_std == label).to_numpy()
        class_matches = matches_array[class_mask]
        class_accuracy = float(class_matches.mean()) if class_matches.size > 0 else 0.0
        support = int(class_support[label])
        class_breakdown.append({
            'label': label,
            'support': support,
            'accuracy': class_accuracy,
            'weight_per_sample': 1.0 / support if support > 0 else 0.0,
        })
    
    if return_details:
        # Create detailed report
        reason_counts = pd.Series(match_reasons).value_counts()
        
        report = {
            'accuracy': accuracy,
            'weighted_accuracy': accuracy,
            'raw_accuracy': raw_accuracy,
            'weighting_scheme': 'balanced_by_ground_truth_class',
            'total_cells': len(matches),
            'correct': sum(matches),
            'incorrect': len(matches) - sum(matches),
            'reason_breakdown': reason_counts.to_dict(),
            'class_breakdown': class_breakdown,
            'matches': matches,
            'match_reasons': match_reasons
        }
        return accuracy, report
    
    return accuracy, None


def generate_accuracy_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    dataset_name: str = 'Dataset',
    allowed_celltypes_only: bool = True,
    myeloid_group: bool = True,
    fine_grained_tolerance: bool = True,
    epithelial_tolerance: bool = True,
) -> str:
    """
    Generate a formatted accuracy report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset_name: Name of the dataset
        allowed_celltypes_only: If True, only include allowed cell types
        myeloid_group: If True, treat myeloid family labels as compatible
        fine_grained_tolerance: If True, allow configured fine-grained tolerance
        epithelial_tolerance: If True, allow epithelial tolerance matching
        
    Returns:
        Formatted report string
    """
    weighted_accuracy, report = calculate_accuracy(
        y_true, y_pred, 
        myeloid_group=myeloid_group,
        fine_grained_tolerance=fine_grained_tolerance,
        epithelial_tolerance=epithelial_tolerance,
        allowed_celltypes_only=allowed_celltypes_only,
        return_details=True
    )
    
    lines = [
        f"=" * 60,
        f"Accuracy Report: {dataset_name}",
        f"=" * 60,
        f"",
        f"Weighted Accuracy: {weighted_accuracy:.4f} ({weighted_accuracy*100:.2f}%)",
        f"Average Accuracy: {report['raw_accuracy']:.4f} ({report['raw_accuracy']*100:.2f}%)",
        f"Weighting Scheme: balanced by ground-truth class (inverse class frequency)",
        f"",
        f"Total Cells: {report['total_cells']}",
        f"  Correct:   {report['correct']}",
        f"  Incorrect: {report['incorrect']}",
        f"",
        f"Match Reason Breakdown:",
    ]
    
    for reason, count in sorted(report['reason_breakdown'].items()):
        pct = count / report['total_cells'] * 100
        lines.append(f"  {reason}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("Per-Class Accuracy (ground-truth weighted):")
    for class_info in report['class_breakdown']:
        lines.append(
            f"  {class_info['label']}: {class_info['accuracy']:.4f} "
            f"({class_info['accuracy']*100:.2f}%), support={class_info['support']}"
        )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# Column Renaming Functions
# ============================================================================

def rename_celltype_column(adata, old_name: str = 'Celltype_training', 
                          new_name: str = 'Ground Truth') -> None:
    """
    Rename cell type column in AnnData object.
    
    Args:
        adata: AnnData object
        old_name: Old column name
        new_name: New column name
    """
    if old_name in adata.obs.columns and new_name not in adata.obs.columns:
        adata.obs[new_name] = adata.obs[old_name]
        del adata.obs[old_name]
        logger.info(f"Renamed column '{old_name}' to '{new_name}'")


def get_ground_truth_column(adata) -> Optional[str]:
    """
    Get the ground truth column name from AnnData.
    
    Args:
        adata: AnnData object
        
    Returns:
        Column name if found, None otherwise
    """
    if 'Ground Truth' in adata.obs.columns:
        return 'Ground Truth'
    elif 'Celltype_training' in adata.obs.columns:
        return 'Celltype_training'
    else:
        # Try to find similar columns
        for col in adata.obs.columns:
            if 'celltype' in col.lower() or 'cell_type' in col.lower():
                return col
        return None


# ============================================================================
# UMAP Color Palette Functions
# ============================================================================

# Vibrant contrast color palette for UMAP visualizations
# Based on scanpy default_20 - good contrast but not fluorescent
VIBRANT_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#aec7e8',  # Light blue
    '#ffbb78',  # Light orange
    '#98df8a',  # Light green
    '#ff9896',  # Light red
    '#c5b0d5',  # Light purple
    '#c49c94',  # Light brown
    '#f7b6d2',  # Light pink
    '#c7c7c7',  # Light gray
    '#dbdb8d',  # Light olive
    '#9edae5',  # Light cyan
    '#393b79',  # Dark blue
    '#637939',  # Dark green
    '#8c6d31',  # Dark brown
    '#843c39',  # Dark red
    '#7b4173',  # Dark magenta
    '#5254a3',  # Indigo
    '#8ca252',  # Moss green
    '#bd9e39',  # Mustard
    '#ad494a',  # Brick red
    '#a55194',  # Magenta
]

def get_aligned_color_palette(gt_labels: pd.Series, pred_labels: pd.Series,
                              base_palette: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """
    Generate color palettes for ground truth and prediction
    such that shared cell types have the same color.
    Uses vibrant contrast colors for better visualization.
    
    Args:
        gt_labels: Ground Truth labels
        pred_labels: Predicted labels
        base_palette: Optional base color palette (defaults to vibrant palette)
        
    Returns:
        Tuple of (gt_palette, pred_palette)
    """
    if base_palette is None:
        # Use vibrant contrast palette
        base_palette = VIBRANT_PALETTE
    
    # Get unique categories
    gt_categories = set(gt_labels.unique())
    pred_categories = set(pred_labels.unique())
    all_categories = sorted(gt_categories | pred_categories)
    
    # Assign colors to all categories
    color_map = {}
    for i, cat in enumerate(all_categories):
        color_map[cat] = base_palette[i % len(base_palette)]
    
    # Create palettes
    gt_palette = {cat: color_map[cat] for cat in sorted(gt_categories)}
    pred_palette = {cat: color_map[cat] for cat in sorted(pred_categories)}
    
    return gt_palette, pred_palette
