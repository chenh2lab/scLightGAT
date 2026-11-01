from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy import sparse

try:
    import scrublet as scr
except ImportError:  # reported only if Scrublet is used
    scr = None

def _assert_count_matrix(matrix, dataset="dataset", allow_fractional=False):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size > 1_000_000:
        values = values[:: max(1, values.size // 1_000_000)]
    if values.size and (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError(f"{dataset}: raw-count matrix contains negative/non-finite values")
    if (
        values.size
        and not allow_fractional
        and not np.allclose(values, np.rint(values), atol=1e-6)
    ):
        raise ValueError(f"{dataset}: expression matrix is not integer raw counts")


def ensure_counts(adata, dataset="dataset"):
    """Select and preserve one verified raw-count matrix."""
    if "raw_counts" in adata.layers:
        matrix = adata.layers["raw_counts"]
        source = "raw_counts"
    elif "counts" in adata.layers:
        matrix = adata.layers["counts"]
        source = "counts"
    else:
        matrix = adata.X
        source = "X"
    allow_fractional = bool(adata.uns.get("allow_fractional_raw_counts", False))
    _assert_count_matrix(matrix, dataset, allow_fractional=allow_fractional)

    # Existing count layers are already independent matrices.  Re-copying both
    # of them at every stage creates a large transient memory spike, especially
    # for the 288k-cell integrated training set.
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = matrix.copy()
    if "counts" not in adata.layers:
        adata.layers["counts"] = matrix.copy()

    # When X itself is the source it is already independent of the two new
    # layer copies.  Otherwise protect the count layer from in-place
    # normalization by putting a single copy in X.
    if source != "X":
        adata.X = matrix.copy()
    return adata


def basic_qc_metrics(adata):
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    percent_top = [n for n in (50, 100, 200, 500) if n < adata.n_vars]
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True,
        percent_top=percent_top or None,
    )
    return adata


def percentile_qc(
    adata,
    low=2,
    high=98,
    mt_max=10.0,
    min_genes_per_cell=200,
    min_cells_per_gene=3,
):
    """Apply the thesis/public-code QC sequence to a count matrix.

    Order is intentionally explicit:
      1) cells >= 200 detected genes
      2) genes detected in >= 3 cells
      3) unique obs/var names
      4) remove all RP* genes
      5) calculate MT/QC metrics on the RP-filtered matrix
      6) strict 2nd/98th percentile filtering for total counts and genes
      7) retain cells with mitochondrial fraction < 10%

    The percentile cutoffs are strict (>, <), matching the public
    ``preprocess_adata`` implementation rather than pandas ``between``.
    """
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # The thesis/public code describes genes beginning with the prefix "RP".
    # Do this before QC metrics so RP expression cannot influence percentile
    # thresholds or the mitochondrial denominator.
    ribosomal = adata.var_names.str.upper().str.startswith("RP")
    adata = adata[:, ~ribosomal].copy()

    basic_qc_metrics(adata)

    total_low, total_high = np.percentile(
        adata.obs["total_counts"], [low, high]
    )
    genes_low, genes_high = np.percentile(
        adata.obs["n_genes_by_counts"], [low, high]
    )

    keep = (
        (adata.obs["total_counts"] > total_low)
        & (adata.obs["total_counts"] < total_high)
        & (adata.obs["n_genes_by_counts"] > genes_low)
        & (adata.obs["n_genes_by_counts"] < genes_high)
        & (adata.obs["pct_counts_mt"] < mt_max)
    )
    adata = adata[keep].copy()

    metadata = {
        "min_genes_per_cell": int(min_genes_per_cell),
        "min_cells_per_gene": int(min_cells_per_gene),
        "ribosomal_prefix_removed": "RP",
        "percentile_low": float(low),
        "percentile_high": float(high),
        "total_low": float(total_low),
        "total_high": float(total_high),
        "genes_low": float(genes_low),
        "genes_high": float(genes_high),
        "mt_max": float(mt_max),
        "percentile_bounds": "strict",
    }
    return adata, metadata


def scdiffusion_qc(adata, min_genes_per_cell=10, min_cells_per_gene=3):
    """Apply the filtering stated by scDiffusion, without thesis percentiles."""
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    basic_qc_metrics(adata)
    return adata, {
        "policy": "scDiffusion source filtering",
        "min_genes_per_cell": int(min_genes_per_cell),
        "min_cells_per_gene": int(min_cells_per_gene),
    }


def run_scrublet(
    adata,
    expected_doublet_rate=0.06,
    seed=0,
    remove=False,
):
    """Run Scrublet on counts using the public scLightGAT parameters."""
    if scr is None:
        raise RuntimeError(
            "scrublet is not installed; recreate the Conda environment from "
            "requirements.txt"
        )

    matrix = adata.layers.get("raw_counts", adata.layers.get("counts", adata.X))
    model = scr.Scrublet(
        matrix,
        expected_doublet_rate=expected_doublet_rate,
        random_state=seed,
    )
    scores, predictions = model.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=30,
    )
    adata.obs["doublet_scores"] = scores
    adata.obs["predicted_doublets"] = predictions.astype(bool)
    adata.obs["doublet_class"] = np.where(
        predictions,
        "Doublets",
        "Singlets",
    )

    if remove:
        adata = adata[~adata.obs["predicted_doublets"]].copy()
    return adata


def normalize_log(adata, target_sum=1e4):
    """Normalize library size and log1p-transform without feature selection.

    Feature engineering (HVG/DGE/Harmony) belongs to scLightGAT, not to this
    source-reconstruction pipeline.
    """
    ensure_counts(adata)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers["log_transformed"] = adata.X.copy()
    # AnnData.raw retains X/var only.  Assigning the completed AnnData directly
    # avoids first copying every large count layer into a temporary AnnData.
    adata.raw = adata
    return adata

def canonical_gene_identifier(value) -> str:
    """Canonicalize non-biological formatting differences in gene IDs.

    This intentionally does *not* translate between identifier systems (for
    example Ensembl -> HGNC).  It only removes serialization artifacts that
    should not change gene identity: surrounding quotes/whitespace, case for
    symbols, and Ensembl version suffixes such as ENSG000001.12.
    """
    import re

    text = str(value).strip()
    # R/write.table-style quoted row names are common in GEO wide matrices.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    upper = text.upper()
    if re.fullmatch(r"ENS[A-Z]*G\d+(?:\.\d+)?", upper):
        upper = re.sub(r"\.\d+$", "", upper)
    return upper


def match_feature_vocabulary(adata, requested_genes, dataset: str):
    """Project ``adata`` to a requested feature vocabulary safely.

    Returns a copied AnnData whose ``var_names`` are the requested/training
    names in requested order.  Exact matches are preferred.  Remaining genes
    may match only through non-biological identifier normalization (quotes,
    case, Ensembl version suffix).  Ambiguous canonical IDs are never guessed.
    """
    import pandas as pd

    requested = [str(gene).strip() for gene in requested_genes if str(gene).strip()]
    source_names = [str(gene) for gene in adata.var_names]
    source_set = set(source_names)

    selected_source = []
    selected_target = []
    exact = 0
    canonical = 0

    canonical_sources = {}
    ambiguous = set()
    for name in source_names:
        key = canonical_gene_identifier(name)
        if key in canonical_sources and canonical_sources[key] != name:
            ambiguous.add(key)
        else:
            canonical_sources[key] = name
    for key in ambiguous:
        canonical_sources.pop(key, None)

    seen_source = set()
    for target in requested:
        if target in source_set:
            source = target
            exact += 1
        else:
            source = canonical_sources.get(canonical_gene_identifier(target))
            if source is not None:
                canonical += 1

        if source is None or source in seen_source:
            continue
        selected_source.append(source)
        selected_target.append(target)
        seen_source.add(source)

    if not selected_source:
        source_examples = [repr(x) for x in source_names[:12]]
        requested_examples = [repr(x) for x in requested[:12]]
        source_keys = [canonical_gene_identifier(x) for x in source_names[:12]]
        requested_keys = [canonical_gene_identifier(x) for x in requested[:12]]
        raise RuntimeError(
            f"{dataset}: none of the requested feature genes are present even "
            "after safe identifier normalization. "
            f"requested examples={requested_examples}; source examples={source_examples}; "
            f"requested canonical={requested_keys}; source canonical={source_keys}. "
            "This likely indicates a true gene-identifier-system mismatch "
            "(for example HGNC symbols vs Ensembl IDs), which must be mapped "
            "explicitly rather than guessed."
        )

    out = adata[:, selected_source].copy()
    out.var["source_feature_name"] = pd.Index(selected_source, dtype=str).to_numpy()
    out.var_names = pd.Index(selected_target, dtype=str)

    print(
        f"[INFO] {dataset} feature projection: {len(selected_target):,}/"
        f"{len(requested):,} requested genes matched "
        f"({exact:,} exact, {canonical:,} canonicalized)",
        flush=True,
    )
    return out
