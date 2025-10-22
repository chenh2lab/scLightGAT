import scanpy as sc
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import harmonypy as hm
import pandas as pd
import logging
import scanpy.external as sce
import scrublet as scr
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

def detect_doublets(adata, expected_doublet_rate=0.06):
    logger.info("💧Starting doublets detection")
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)
    
    adata.obs['doublet_scores'], adata.obs['predicted_doublets'] = scrub.scrub_doublets(
        min_counts=2, 
        min_cells=3, 
        min_gene_variability_pctl=85, 
        n_prin_comps=30
    )
    
    adata.obs['predicted_doublets'] = adata.obs['predicted_doublets'].map({True: 'Doublets', False: 'Singlets'})
    
    logger.info(f"💧🔍Doublet detection results: {adata.obs['predicted_doublets'].value_counts()}")
    logger.info("✅Doublet detection completed")
    return adata

def preprocess_adata(adata, upper_lim=98, lower_lim=2, use_hvg=False, doublet_detect=True, keep_doublets=True, expected_doublet_rate=0.06):
    logger.info("Starting preprocessing")
    
    logger.info("🧬Filtering cells and genes")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.raw = adata.copy()

    logger.info("🧬Filtering out specific RP & MT genes and calculating their QC metrics")
    genes_to_keep = [gene for gene in adata.var_names if not gene.startswith('RP')]
    adata = adata[:, genes_to_keep].copy()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    logger.info("👽Filtering outliers")
    lower_count, upper_count = np.percentile(adata.obs.total_counts, [lower_lim, upper_lim])
    lower_gene, upper_gene = np.percentile(adata.obs.n_genes_by_counts, [lower_lim, upper_lim])
    adata = adata[(adata.obs.total_counts > lower_count) & 
                  (adata.obs.total_counts < upper_count) &
                  (adata.obs.n_genes_by_counts > lower_gene) &
                  (adata.obs.n_genes_by_counts < upper_gene)]
    
    if doublet_detect:   
        adata = detect_doublets(adata, expected_doublet_rate)
        if not keep_doublets:
            logger.info("Filtering out doublets")
            adata = adata[adata.obs["predicted_doublets"] == "Singlets"]

    logger.info("Normalizing and log-transforming data")
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log_transformed"] = adata.X.copy()

    if use_hvg:
        logger.info("🧬Selecting highly variable genes")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer="log_transformed", flavor="seurat_v3", batch_key='Celltype_training')
    
    logger.info("Store condensed-scRNA matrix")
    adata.X = scipy.sparse.csr_matrix(adata.X)
    
    logger.info("✅Preprocessing completed")
    return adata

def visualization_process(adata, res=1, batch_key=None, min_cluster_size=50):
    logger.info("Starting visualization process")

    # Always ensure PCA exists as a fallback
    logger.info("Performing PCA (default 50 comps)")
    sc.pp.pca(adata, n_comps=50)

    # Decide which embedding to use for neighbors/UMAP:
    # Priority:
    # 1) If batch_key is provided -> compute Harmony now and set X_pca_harmony
    # 2) Else if precomputed X_pca_harmony exists -> prefer it
    # 3) Else fall back to X_pca
    use_rep = 'X_pca'
    if batch_key:
        logger.info(f"🔧Performing batch correction via Harmony with batch key: {batch_key}")
        data_mat = adata.obsm['X_pca']
        meta_data = adata.obs
        ho = hm.run_harmony(data_mat, meta_data, [batch_key], max_iter_harmony=50, plot_convergence=True)
        adata.obsm['X_pca_harmony'] = pd.DataFrame(ho.Z_corr).T.values
        use_rep = 'X_pca_harmony'
    elif 'X_pca_harmony' in adata.obsm:
        logger.info("Detected precomputed X_pca_harmony; using it for neighbors/UMAP")
        use_rep = 'X_pca_harmony'

    logger.info(f"Computing neighbors (use_rep={use_rep})")
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15, metric='cosine')

    logger.info("Computing UMAP")
    sc.tl.umap(adata)

    logger.info("Performing Leiden clustering")
    sc.tl.leiden(adata, resolution=res)

    # Optional: placeholder for merging tiny clusters (no-op)
    cluster_sizes = adata.obs['leiden'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
    if len(small_clusters) > 0:
        logger.info(f"Merging {len(small_clusters)} small clusters (placeholder)")
        # add your merge strategy if needed

    logger.info("✅Visualization process completed")
    return adata



def main_process(
    adata, 
    upper_lim=98, 
    lower_lim=2, 
    use_hvg=False, 
    doublet_detect=True, 
    keep_doublets=True, 
    expected_doublet_rate=0.06, 
    batch_key=None, 
    res=0.8, 
    visualize=True,
    output_dir="output_plots"
):
    """
    Main processing pipeline for scRNA-seq data.

    Parameters:
    - adata: AnnData object
    - upper_lim: Upper percentile for outlier removal
    - lower_lim: Lower percentile for outlier removal
    - use_hvg: Whether to use highly variable genes
    - doublet_detect: Whether to perform doublet detection
    - keep_doublets: Whether to retain doublets after detection
    - expected_doublet_rate: Expected doublet rate for scrublet
    - batch_key: Key for batch correction (optional)
    - res: Resolution for Leiden clustering
    - visualize: Whether to generate visualizations
    - output_dir: Directory to save visualizations

    Returns:
    - Processed AnnData object
    """
    logger.info("💫💫💫Starting main processing pipeline💫💫💫")
    
    # Step 1: Preprocess the data
    adata = preprocess_adata(
        adata, 
        upper_lim, 
        lower_lim, 
        use_hvg, 
        doublet_detect, 
        keep_doublets, 
        expected_doublet_rate
    )
    
    # # Step 2: Visualize and cluster the data
    # adata = visualization_process(
    #     adata, 
    #     res=res, 
    #     batch_key=batch_key, 
    #     min_cluster_size=50
    # )
    
    # # Step 3: Generate visualizations if enabled
    # if visualize:
    #     generate_visualizations(adata, output_dir)
    
    logger.info("💫💫💫Main processing pipeline completed💫💫💫")
    return adata


def generate_visualizations(adata, output_dir="output_plots"):
    """
    Generate QC and clustering visualizations.
    """
    logger.info("📊Generating visualizations")
    
    # Ensure the output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Jointplot visualization
    sns.jointplot(
        data=adata.obs,
        x="log1p_total_counts",
        y="log1p_n_genes_by_counts",
        kind="hex",
        color="blue"
    )
    plt.title("Training Dataset: Total Counts vs Genes by Counts")
    jointplot_path = f"{output_dir}/jointplot_counts_genes.png"
    plt.savefig(jointplot_path, dpi=300)
    logger.info(f"Jointplot saved to {jointplot_path}")
    plt.show()
    
    # # Highly variable genes visualization
    # sc.pl.highly_variable_genes(adata, save=f"_hvg_{output_dir}.png")
    # logger.info("Highly variable genes plot generated")
    
    # UMAP visualization if available
    if "X_umap" in adata.obsm.keys():
        sc.pl.umap(adata, color=["leiden"], save=f"_umap_clusters_{output_dir}.png")
        logger.info("UMAP plot generated")
    
    logger.info("✅All visualizations completed")
