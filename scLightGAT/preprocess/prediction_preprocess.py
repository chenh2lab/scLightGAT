import scanpy as sc
import pandas as pd
import numpy as np
import itertools
import logging
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx
from torch_geometric.data import Data
from pathlib import Path
from scLightGAT.preprocess.preprocess import preprocess_adata

import os

from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)


def get_celltype_column(adata, celltype_col=None):
    """Determine the celltype column to use."""
    if celltype_col is not None and celltype_col in adata.obs.columns:
        return celltype_col
    elif "Celltype_training" in adata.obs.columns:
        return "Celltype_training"
    else:
        available_columns = ", ".join(adata.obs.columns)
        raise ValueError(f"No valid celltype column found. Available columns are: {available_columns}")

def feature_selection(adata, group):
    logger.info(f"Performing feature selection using {group}")
    sc.tl.rank_genes_groups(adata, groupby=group, method='t-test', reference='rest', use_raw=True)
    markers = sc.get.rank_genes_groups_df(adata, None)
    
    markers_filtered = markers[(markers.pvals_adj < 0.01) & (markers.logfoldchanges > 2)]
      
    markers_sorted = markers_filtered.sort_values("scores", ascending=False).reset_index(drop=True)
    markers_sorted['z_score_rank'] = markers_sorted['scores'].rank(ascending=False)
    markers_sorted['logFC_rank'] = markers_sorted['logfoldchanges'].abs().rank(ascending=False)
    markers_sorted['combined_rank'] = (markers_sorted['z_score_rank'] + markers_sorted['logFC_rank']) / 2
    markers_sorted = markers_sorted.sort_values('combined_rank', ascending=True).reset_index(drop=True)
    
    return markers_sorted

def generate_featurelist(markers_sorted, top):
    logger.info(f"Generating feature list with top {top} features")
    selected_features = set()
    final_features_per_cell_type = {group: [] for group in markers_sorted['group'].unique()}
    
    top_features = markers_sorted[~markers_sorted['names'].isin(selected_features)].head(top)
    selected_features.update(top_features['names'])
    
    for group in final_features_per_cell_type.keys():
        group_features = top_features[top_features['group'] == group]['names'].tolist()
        
        if len(group_features) < 50:
            additional_features = markers_sorted[
                (markers_sorted['group'] == group) & (~markers_sorted['names'].isin(selected_features))
            ]['names'].tolist()
            features_to_add = additional_features[:max(0, 50 - len(group_features))]
            group_features.extend(features_to_add)
            selected_features.update(features_to_add)
        
        final_features_per_cell_type[group] = group_features[:100]
    
    return final_features_per_cell_type

def prepare_data_for_prediction(adata_train, adata_test, celltype_col=None, top_features=3000):
    logger.info("⏰Preparing DGEs data")
    group = get_celltype_column(adata_train, celltype_col)
    train_markers = feature_selection(adata_train, group)
    train_dict = generate_featurelist(train_markers, top_features)

    train_deglist = list(itertools.chain(*train_dict.values()))
    common_hvgs = list(set(train_deglist) & set(adata_train.var_names) & set(adata_test.var_names))

    logger.info(f"🧬Number of common DGEs: {len(common_hvgs)}🧬")

    adata_train_dge = adata_train[:, common_hvgs].copy()
    adata_test_dge = adata_test[:, common_hvgs].copy()

    logger.info(f"Final shape of adata_train_dge: {adata_train_dge.shape}")
    logger.info(f"Final shape of adata_test_dge: {adata_test_dge.shape}")
    adata_train_dge.var['prominent'] = adata_train_dge.var_names.isin(common_hvgs)


    return adata_train_dge, adata_test_dge, common_hvgs

def prepare_hvg_data(adata_train, adata_test, celltype_col=None):
    logger.info("⏰Preparing HVG data")
    group = get_celltype_column(adata_train, celltype_col)
    sc.pp.highly_variable_genes(adata_train, n_top_genes=3000, subset=False, layer="log_transformed", flavor="seurat_v3", batch_key=group)
    hvgs = adata_train.var[adata_train.var['highly_variable']].index
    
    # common HVGs
    common_hvgs = list(set(hvgs) & set(adata_test.var_names))
    logger.info(f"🧬Number of common HVGs: {len(common_hvgs)}🧬")

    adata_train_hvg = adata_train[:, common_hvgs].copy()
    adata_test_hvg = adata_test[:, common_hvgs].copy()

    logger.info(f"Final shape of adata_train_hvg: {adata_train_hvg.shape}")
    logger.info(f"Final shape of adata_test_hvg: {adata_test_hvg.shape}")

    return adata_train_hvg, adata_test_hvg

def transform_adata(adata, check_type):
    logger.info(f"Transforming AnnData object for {check_type}")
    if "log_transformed" not in adata.layers:
        adata.layers["log_transformed"] = np.log1p(adata.X)
    matrix = adata.to_df(layer="log_transformed")
    adata_pd = pd.DataFrame(adata.obs)
    matrix["Cell_type"] = adata_pd[check_type]
    return matrix

def prepare_lightgbm_data(adata_dge, celltype_col=None):
    logger.info("💾Preparing data for LightGBM")
    check_type = get_celltype_column(adata_dge, celltype_col)
    matrix = transform_adata(adata_dge, check_type)
    # remove duplicated genes
    matrix = matrix.loc[:,~matrix.columns.duplicated()]
    
    encoder = LabelEncoder()
    matrix['encoding'] = encoder.fit_transform(matrix["Cell_type"])
    X = matrix.drop(["Cell_type", 'encoding'], axis=1)
    X.columns = X.columns.astype(str)
    y = matrix['encoding']
    
    return X, y, encoder

def prepare_dvae_data(adata_hvg, batch_size=32, celltype_col=None):
    logger.info("💾Preparing data for DVAE")
    check_type = get_celltype_column(adata_hvg, celltype_col)
    matrix = transform_adata(adata_hvg, check_type)
    
    encoder = LabelEncoder()
    matrix['encoding'] = encoder.fit_transform(matrix["Cell_type"])
    
    X = matrix.drop(["Cell_type", 'encoding'], axis=1)
    y = matrix['encoding']
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X.shape[1]
    
    return X, y, input_dim, dataloader, encoder

def balance_classes(X, y, target_count=10000):
    logger.info(f"Balancing classes with target count {target_count}")
    class_counts = pd.Series(y).value_counts()
    logger.info(f"Original class counts:\n{class_counts}")

    over_strategy = {label: target_count for label, count in class_counts.items() if count < target_count}
    under_strategy = {label: target_count for label, count in class_counts.items() if count > target_count}

    steps = []
    if over_strategy:
        steps.append(('over', RandomOverSampler(sampling_strategy=over_strategy)))
    if under_strategy:
        steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy)))

    pipeline = Pipeline(steps=steps)
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    logger.info(f"🔧Resampled class counts:\n{pd.Series(y_resampled).value_counts()}")
    return X_resampled, y_resampled

def prepare_data(adata_train, adata_test, balanced_counts=10000, batch_size=32, celltype_col=None):
    logger.info("💾Preparing data for all models")
    
    for adata in [adata_train, adata_test]:
        if "log_transformed" not in adata.layers:
            adata.layers["log_transformed"] = np.log1p(adata.X)

    
    # Prepare DGE and HVG data
    adata_train_dge, adata_test_dge, common_hvgs = prepare_data_for_prediction(adata_train, adata_test, celltype_col)
    adata_train_hvg, adata_test_hvg = prepare_hvg_data(adata_train, adata_test, celltype_col)
    
    for adata in [adata_train_dge, adata_test_dge, adata_train_hvg, adata_test_hvg]:
        if "log_transformed" not in adata.layers:
            adata.layers["log_transformed"] = np.log1p(adata.X)
    
    # Prepare data for LightGBM (using DGE data)
    X_train, y_train, encoder_lightgbm = prepare_lightgbm_data(adata_train_dge, celltype_col)
    
    # Prepare data for DVAE (using HVG data)
    X_hvg, y_hvg, input_dim, dataloader, encoder_dvae = prepare_dvae_data(adata_train_hvg, batch_size, celltype_col)
    
    # Use the same resampling for both LightGBM and DVAE
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, balanced_counts)
    
    # Create a new dataloader with balanced data for DVAE
    balanced_dataset = TensorDataset(torch.tensor(X_hvg.values, dtype=torch.float32))
    balanced_dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    
    return {
        'lightgbm': (X_train_balanced, y_train_balanced, encoder_lightgbm),
        'dvae': (X_hvg, y_hvg, input_dim, balanced_dataloader, encoder_dvae),
        'adata_train_dge': adata_train_dge,
        'adata_test_dge': adata_test_dge,
        'adata_train_hvg': adata_train_hvg,
        'adata_test_hvg': adata_test_hvg,
        'common_hvgs': common_hvgs
    }

def main_data_preparation(adata_train, adata_test, balanced_counts=10000, batch_size=32, celltype_col=None,
                          preprocess_params=None):
    logger.info("💫💫💫Starting main data preparation process💫💫💫")
    
    # Preprocess adata_train and adata_test if preprocess_params is provided
    if preprocess_params is not None:
        adata_train = preprocess_adata(adata_train, **preprocess_params)
        adata_test = preprocess_adata(adata_test, **preprocess_params)
    
    # Prepare data for all models
    all_data = prepare_data(adata_train, adata_test, balanced_counts, batch_size, celltype_col)
 
    logger.info("💫💫💫Data preparation completed💫💫💫")
    return all_data



def split_and_save_cell_groups(adata, output_dir='group_data'):

    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # define
    cell_groups = {
        'lymphoid': ['CD4+T cells', 'CD8+T cells', 'NK cells'],
        'b_lineage': ['pDC', 'B cells', 'Plasma cells'],
        'myeloid': ['DC', 'Macrophages', 'Monocytes'],
        'others': ['Epithelial cells', 'Endothelial cells', 'Fibroblasts', 'Mast cells']
    }
    
    
    group_adatas = {}
    for group_name, cell_types in cell_groups.items():
        
        mask = adata.obs['Celltype_training'].isin(cell_types)
        
        
        group_adata = adata[mask].copy()
        
        
        cell_counts = group_adata.obs['Celltype_training'].value_counts()
        print(f"\n{group_name} group cell counts:")
        print(cell_counts)
        print(f"Total cells: {len(group_adata)}")
        
        
        output_file = os.path.join(output_dir, f'{group_name}_group.h5ad')
        group_adata.write_h5ad(output_file)
        print(f"Saved to {output_file}")
        
        
        group_adatas[group_name] = group_adata
    
    
    return group_adatas
def prepare_subtype_data_from_gat(adata_train, gat_results, broad_type, top_features=1000):
    """According GAT outcome to predict second round"""
    logger.info(f"Preparing subtype data for {broad_type}")
    
    # Define cell subtype
    broad_to_subtypes = {
        'CD4+T cells': ['CD4+Tfh/Th cells', 'CD4+exhausted T cells', 'CD4+memory T cells',
                       'CD4+naive T cells', 'CD4+reg T cells'],
        'CD8+T cells': ['CD8+MAIT T cells', 'CD8+Naive T cells', 'CD8+exhausted T cells',
                       'CD8+memory T cells'],
        'B cells': ['Follicular B cells', 'Germinal B cells', 'MALT B cells',
                   'Memory B cells', 'Naive B cells'],
        'DC': ['cDC', 'DC'],
        'Plasma cells': ['IgA+ Plasma', 'IgG+ Plasma', 'Plasma cells', 'Plasmablasts']
    }
    
    
    test_mask = gat_results.obs['GAT_pred'] == broad_type
    train_mask = adata_train.obs['Celltype_training'] == broad_type
    
    group_train = adata_train[train_mask].copy()
    group_test = gat_results[test_mask].copy()
    
    if len(group_train) == 0 or len(group_test) == 0:
        logger.warning(f"No data for {broad_type}")
        return None, None, test_mask, None

    if broad_type not in broad_to_subtypes:
        logger.info(f"{broad_type} has no subtypes")
        return None, None, test_mask, None
    
    # DGE analysis for feature selection
    train_markers = feature_selection(group_train, 'Celltype_subtraining')
    train_dict = generate_featurelist(train_markers, top_features)
    
    dge_genes = list(itertools.chain(*train_dict.values()))
    common_genes = list(set(dge_genes) & set(gat_results.var_names))
    
    group_train = group_train[:, common_genes].copy()
    group_test = group_test[:, common_genes].copy()
    
    return group_train, group_test, test_mask, broad_to_subtypes[broad_type]

