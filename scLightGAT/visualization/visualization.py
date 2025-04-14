import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics import confusion_matrix
from scLightGAT.logger_config import setup_logger
from typing import List, Optional, Dict, Tuple, Any


logger = setup_logger(__name__)

def check_visualization_ready(adata):
    """
    Check if the AnnData object has UMAP coordinates for visualization.
    
    Args:
        adata: AnnData object to check
        
    Returns:
        bool: True if UMAP coordinates exist, False otherwise
    """
    if 'X_umap' not in adata.obsm:
        logger.warning("UMAP coordinates not found. Please run pre_visualization_process first.")
        return False
    return True

def improved_barplot(adata, column, title, figsize=(14, 7)):
    """
    Create an improved horizontal barplot for visualizing distributions in AnnData.
    
    Args:
        adata: AnnData object containing the data
        column: Column name in adata.obs to plot
        title: Title for the plot
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    logger.info(f"Creating improved barplot for {column}")
    data = adata.obs[column].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bars with viridis color palette
    bars = ax.barh(data.index, data.values, color=sns.color_palette("viridis", len(data)))
    
    # Add value labels at the end of each bar
    for i, v in enumerate(data.values):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
    
    # Customize plot appearance
    ax.set_title(f'{title} Distribution', fontsize=20, pad=20)
    ax.set_xlabel('Count', fontsize=14)
    ax.set_ylabel(title, fontsize=14)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
def plot_umap_by_label(adata, label: str, save_path: str = None):
    """
    Plot UMAP colored by a categorical label and optionally save the figure.

    Args:
        adata: AnnData object with UMAP computed.
        label: Key in adata.obs to color the UMAP plot.
        save_path: Path to save the plot image (optional).
    """
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    plt.figure(figsize=(8, 6))
    sc.pl.umap(
        adata,
        color=label,
        show=False,
        frameon=False,
        title=f'UMAP colored by {label}'
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_doublet_comparison(adata):
    """
    Create a comparison plot showing doublet predictions and scores.
    
    Args:
        adata: AnnData object containing doublet information
        
    Returns:
        None
    """
    if not check_visualization_ready(adata):
        return

    logger.info("Creating doublet comparison plot")
    
    # Calculate statistics
    doublets_count = sum(adata.obs['predicted_doublets'] == 'Doublets')
    singlets_count = sum(adata.obs['predicted_doublets'] == 'Singlets')
    total_count = len(adata.obs['predicted_doublets'])

    stats_text = (f"Doublets: {doublets_count} ({doublets_count/total_count:.2%})\n"
                 f"Singlets: {singlets_count} ({singlets_count/total_count:.2%})\n"
                 f"Total: {total_count}")

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot doublet predictions
    sc.pl.umap(adata, color='predicted_doublets', title="Doublet Prediction",
               frameon=False, palette={'Doublets': 'red', 'Singlets': 'blue'},
               size=10, ax=ax1, show=False)

    # Plot doublet scores
    sc.pl.umap(adata, color='doublet_scores', title="Doublet Scores",
               frameon=False, cmap='viridis', size=10, ax=ax2, show=False)

    # Customize titles and add statistics
    ax1.set_title("Doublet Prediction", fontsize=14, fontweight='bold')
    ax2.set_title("Doublet Scores", fontsize=14, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('doublet_comparison_umap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_comparison(adata, prediction_columns, figsize=(45, 20)):
    """
    Create a multi-panel plot comparing different prediction methods.
    
    Args:
        adata: AnnData object containing the data
        prediction_columns: List of column names to compare
        figsize: Figure size tuple (width, height)
    """
    if not check_visualization_ready(adata):
        return

    logger.info("Creating prediction comparison plot")
    
    # Calculate number of rows needed
    n_cols = 3
    n_rows = (len(prediction_columns) + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    
    color_map = dict(zip(
        sorted(set.union(*[set(adata.obs[col].astype('category').cat.categories) for col in prediction_columns])),
        sns.color_palette("husl", n_colors=20) +  
        sns.color_palette("Set2", n_colors=8) +
        sns.color_palette("Paired", n_colors=12)
    ))

    title_fontsize = 20
    legend_fontsize = 12 
    
    for idx, col in enumerate(prediction_columns):
        row = idx // n_cols
        col_idx = idx % n_cols
        
        try:
            sc.pl.umap(adata, 
                      color=col, 
                      ax=axs[row, col_idx], 
                      title=col,
                      frameon=False, 
                      legend_loc='right margin', 
                      show=False,
                      size=15, 
                      alpha = 0.8,
                      palette=color_map,
                      legend_fontsize=legend_fontsize)  # Larger legend font
            
            # Enhance title appearance
            axs[row, col_idx].set_title(col, 
                                      fontsize=title_fontsize, 
                                      fontweight='bold',
                                      pad=20)  # Add padding to title
            
        except Exception as e:
            logger.error(f"Error plotting {col}: {e}")
            axs[row, col_idx].text(0.5, 0.5, 
                                 f"Error plotting {col}: {e}",
                                 ha='center', 
                                 va='center')
    
    # Remove empty subplots
    for idx in range(len(prediction_columns), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        fig.delaxes(axs[row, col_idx])

    plt.suptitle("Cell Type Prediction Comparison", 
                 fontsize=24,  # Even larger font for main title
                 fontweight='bold',
                 y=1.02)  # Adjust title position
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.5)  # Add padding around the figure
    plt.show()

def plot_group_accuracy(adata, prediction_column, group_mappings, reference_column='Manual Annotation'):
    """
    Calculate and plot accuracy for different cell type groups.
    
    Args:
        adata: AnnData object containing the data
        prediction_column: Column name containing predictions
        group_mappings: Dictionary mapping cell types to their groups
        reference_column: Column name containing ground truth labels
        
    Returns:
        None
    """
    logger.info("Creating group accuracy comparison plot")
    
    accuracies = {}
    counts = {}
    
    # Calculate accuracy for each group
    for group_name, cell_types in group_mappings.items():
        mask = adata.obs[reference_column].isin(cell_types)
        if sum(mask) > 0:
            group_true = adata.obs[reference_column][mask]
            group_pred = adata.obs[prediction_column][mask]
            
            # Count correct predictions considering group membership
            correct = sum([(true in cell_types and pred in cell_types)
                         for true, pred in zip(group_true, group_pred)])
            total = sum(mask)
            
            accuracies[group_name] = correct / total
            counts[group_name] = total

    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values())
    
    # Add value labels and counts
    for bar, count in zip(bars, counts.values()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}\n(n={count})',
                ha='center', va='bottom')

    plt.title('Prediction Accuracy by Cell Type Group', fontsize=14, fontweight='bold')
    plt.xlabel('Cell Type Group', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('group_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_losses(epoch_losses, model_name="Model", figsize=(10, 6), save_path=None):
    """
    Plot training losses over epochs.
    
    Args:
        epoch_losses: List of loss values for each epoch
        model_name: Name of the model for plot title
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    logger.info(f"Creating training loss plot for {model_name}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the loss curve
    epochs = range(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses, 'b-', linewidth=2, label=f'{model_name} Loss')
    
    # Add points for easier reading
    ax.plot(epochs, epoch_losses, 'b.', markersize=8)
    
    # Customize plot appearance
    ax.set_title(f'{model_name} Training Loss', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add final loss value annotation
    final_loss = epoch_losses[-1]
    ax.text(0.98, 0.02, f'Final Loss: {final_loss:.4f}',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig




def create_all_visualizations(adata, prediction_columns, group_mappings=None, dvae_losses=None, gat_losses=None):
    """
    Create all visualization plots for the analysis.
    
    Args:
        adata: AnnData object containing the data
        prediction_columns: List of columns to compare in predictions
        group_mappings: Optional dictionary mapping cell types to groups
        dvae_losses: Optional list of DVAE training losses
        gat_losses: Optional list of GAT training losses
        
    Returns:
        None
    """
    logger.info("Creating all visualizations")
    
    # Create basic distribution plot
    improved_barplot(adata, 'Celltype_training', 'Cell Type')
    
    # Create doublet comparison plot
    plot_doublet_comparison(adata)
    
    # Create prediction comparison plot
    plot_prediction_comparison(adata, prediction_columns)
    
    # Create confusion matrix
    if 'Manual Annotation' in adata.obs and 'LightGBMprediction' in adata.obs:
        plot_confusion_matrix(
            adata.obs['Manual Annotation'],
            adata.obs['LightGBMprediction'],
            class_names=adata.obs['Manual Annotation'].unique(),
            title="LightGBM Prediction Confusion Matrix"
        )
    
    # Create group accuracy plot if group mappings are provided
    if group_mappings is not None:
        plot_group_accuracy(adata, 'LightGBMprediction', group_mappings)
    
    # Plot training losses if provided
    if dvae_losses is not None:
        plot_training_losses(
            dvae_losses,
            model_name="DVAE",
            save_path="dvae_training_loss.png"
        )
    
    if gat_losses is not None:
        plot_training_losses(
            gat_losses,
            model_name="GAT",
            save_path="gat_training_loss.png"
        )
    
    logger.info("All visualizations created")
    
    
def create_dvae_visualizations(latent_features: np.ndarray,
                             cell_types: np.ndarray, 
                             title: str,
                             dvae_losses: List[float],
                             save_path: Optional[str] = None):
    """Create visualizations for DVAE results"""
    
    if len(latent_features) != len(cell_types):
        raise ValueError(f"Length mismatch: latent_features ({len(latent_features)}) "
                        f"!= cell_types ({len(cell_types)})")
    
    # UMAP visualization of latent space
    temp_adata = sc.AnnData(latent_features)
    temp_adata.obs['cell_type'] = pd.Categorical(cell_types) 
    
    # Compute UMAP
    sc.pp.neighbors(temp_adata, n_neighbors=10, use_rep='X')
    sc.tl.umap(temp_adata, min_dist=0.1)
    
    # Create subplot with UMAP and loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot UMAP
    sc.pl.umap(
        temp_adata,
        color='cell_type',
        title=f"{title} Latent Space",
        frameon=False,
        legend_loc='right margin',
        legend_fontsize=8,
        size=8,
        alpha=0.7,
        show=False,
        ax=ax1
    )
    
    # Plot loss curve
    epochs = range(1, len(dvae_losses) + 1)
    ax2.plot(epochs, dvae_losses, 'b-', linewidth=2)
    ax2.plot(epochs, dvae_losses, 'b.', markersize=8)
    ax2.set_title('DVAE Training Loss', fontsize=16)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()