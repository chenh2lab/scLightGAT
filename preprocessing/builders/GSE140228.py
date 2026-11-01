from __future__ import annotations

from builders.common import (
    add_data_source,
    best_column,
    combine_same_genes,
    download_geo_dataset,
    find,
    subset_to_metadata,
)
from utils.config import GEO_PATTERNS
from utils.donor import attach_donor_metadata
from utils.io import read_10x_mtx, read_table, read_wide_counts


def _tumor_only(adata):
    # GSE140228 publishes Tissue_sub in both droplet and Smart-seq2 cell-info
    # tables. Prefer that explicit biological field; retain a conservative
    # fallback for mirrors that rename columns.
    column = "Tissue_sub" if "Tissue_sub" in adata.obs.columns else best_column(
        adata.obs,
        [r"tissue.*sub", r"site", r"tissue", r"location", r"sample"],
    )
    if column is None:
        raise RuntimeError(
            "GSE140228: no tissue/site column was found for tumor selection"
        )

    keep = adata.obs[column].astype(str).str.contains(
        "tumor",
        case=False,
        na=False,
    )
    if not keep.any():
        values = sorted(adata.obs[column].dropna().astype(str).unique())[:30]
        raise RuntimeError(
            f"GSE140228: tumor selection using {column!r} matched zero cells; "
            f"observed examples={values}"
        )
    return adata[keep].copy()


def build(rawdir, auxdir):
    source = download_geo_dataset(
        "GSE140228",
        "GSE140228",
        rawdir,
        GEO_PATTERNS,
    )

    # This GEO gene table is not a headerless 10x genes.tsv: it has an
    # ENSEMBL/SYMBOL header row.  Use the published SYMBOL column explicitly.
    droplet = read_10x_mtx(
        find(source, r"Droplet\.mtx"),
        find(source, r"Droplet_barcodes"),
        find(source, r"Droplet_genes"),
        genes_header=0,
        gene_name_column="SYMBOL",
        deduplicate_genes=True,
    )
    droplet_metadata = read_table(
        find(source, r"Droplet_cellinfo"), index_col=0
    )
    droplet_metadata.index = droplet_metadata.index.astype(str)
    # Published loading code uses the first metadata column as row names and
    # verifies that those row names equal the matrix barcodes.
    droplet, matched, source_column = subset_to_metadata(
        droplet,
        droplet_metadata,
    )
    print(
        f"  GSE140228 droplet metadata matching: {matched:,}/{len(droplet_metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )
    droplet = _tumor_only(droplet)

    smartseq = read_wide_counts(find(source, r"read_counts_Smartseq2"), sep=",")
    smartseq_metadata = read_table(
        find(source, r"cell_info_Smartseq2"), index_col=0
    )
    smartseq_metadata.index = smartseq_metadata.index.astype(str)
    smartseq, matched, source_column = subset_to_metadata(
        smartseq,
        smartseq_metadata,
    )
    print(
        f"  GSE140228 Smart-seq2 metadata matching: {matched:,}/{len(smartseq_metadata):,} "
        f"cells using {source_column}",
        flush=True,
    )
    smartseq = _tumor_only(smartseq)

    combined = combine_same_genes([droplet, smartseq])
    # The deposited Smart-seq2 file is named read_counts but contains
    # non-negative fractional assignments (for example 0.33, 0.5 and 0.67).
    # Preserve those published raw values; rounding would alter library sizes.
    combined.uns["allow_fractional_raw_counts"] = True
    combined = add_data_source(combined, "GSE140228")
    return attach_donor_metadata(combined, "GSE140228")
