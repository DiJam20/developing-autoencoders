import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from autoencoder import *
from solver import *
from model_utils import *


def display_ready_rf(rf: np.ndarray, dataset: str = "mnist") -> np.ndarray:
    """
    Reshape the RF to be displayed as an image.

    Args:
        rf (np.ndarray): Receptive field to reshape.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
    
    Returns:
        np.ndarray: Reshaped receptive field.
    """
    if dataset.lower() == "mnist":
        IMG_WIDTH, IMG_HEIGHT = 28, 28
    elif dataset.lower() == "cifar":
        IMG_WIDTH, IMG_HEIGHT = 32, 32
    
    if len(rf.shape) == 1:
        return rf.reshape(IMG_WIDTH, IMG_HEIGHT)
    return rf


def plot_rfs_for_single_model(model_type: str, dataset: str = "mnist", model_idx: int = 0, epoch_idx: int = -1) -> None:
    """
    Plot receptive fields for all neurons in a model at a specific epoch.
    
    Args:
        model_type (str): Type of model ('sae' or 'dae')
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        model_idx (int): Index of the model to visualize
        epoch_idx (int): Index of epoch to visualize (-1 for last epoch)

    Returns:
        None: Plots the receptive fields and saves the figure.
    """
    if dataset.lower() == "mnist":
        IMG_WIDTH, IMG_HEIGHT = 28, 28
        MAX_NEURONS = 32
        fig_size = (20, 9)
        subplot_rows, subplot_cols = 4, 8
    elif dataset.lower() == "cifar":
        IMG_WIDTH, IMG_HEIGHT = 32, 32
        MAX_NEURONS = 128
        fig_size = (20, 15)
        subplot_rows, subplot_cols = 10, 13
    
    rf_matrices = np.load(f"Results/{dataset}_{model_type}_rfs.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
    rf_ls = rf_matrix[epoch_idx]
    
    fig = plt.figure(figsize=fig_size)
    model_name = 'AE' if model_type == 'sae' else 'DevAE'
    fig.suptitle(f"Receptive Fields of All Neurons ({model_name}, {dataset.upper()})", fontsize=24)
    
    for i in range(MAX_NEURONS):
        plt.subplot(subplot_rows, subplot_cols, i+1)
        plt.title(str(i+1))
        plt.axis('off')
        
        # Only plot if the index exists in rf_ls, otherwise leave empty
        if i < len(rf_ls):
            if dataset.lower() == "mnist":
                plt.imshow(display_ready_rf(rf_ls[i], dataset), cmap='gray')
            else:
                plt.imshow(np.transpose(display_ready_rf(rf_ls[i], dataset), (1, 2, 0)))
        else:
            plt.imshow(np.zeros((IMG_WIDTH, IMG_HEIGHT)), cmap='gray')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    epoch_label = epoch_idx if epoch_idx >= 0 else f"final"
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_rfs_model_{model_idx}_epoch_{epoch_label}.png", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_{model_type}_rfs_model_{model_idx}_epoch_{epoch_label}.svg")
    plt.close(fig)


def plot_rf_over_time(model_type: str, dataset: str = "mnist", model_idx: int = 0, neuron_idx: int = 0) -> None:
    """
    Plot the receptive field development of a single neuron over all epochs for a single model.

    Args:
        model_type (str): Type of model to plot RF development for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        model_idx (int): Index of the model to plot RF development for.
        neuron_idx (int): Index of the neuron to plot RF development for.

    Returns:
        None: Plots the receptive field development and saves the figure.
    """
    rf_matrices = np.load(f"Results/{dataset}_{model_type}_rfs.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
        
    # Calculate grid dimensions
    cols = 10
    rows = (len(rf_matrix) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))
    axes = axes.ravel()
    
    for i in range(len(rf_matrix)):
        if dataset.lower() == "mnist":
            axes[i].imshow(display_ready_rf(rf_matrix[i][neuron_idx], dataset), cmap='gray')
        else:
            axes[i].imshow(np.transpose(display_ready_rf(rf_matrix[i][neuron_idx], dataset), (1, 2, 0)))
        axes[i].axis('off')

    model_name = 'AE' if model_type == 'sae' else 'DevAE'
    fig.suptitle(f"Receptive Field Development of Neuron {neuron_idx} ({model_name}, {dataset.upper()})", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_rf_development_model_{model_idx}_neuron_{neuron_idx}.png", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_{model_type}_rf_development_model_{model_idx}_neuron_{neuron_idx}.svg")
    plt.close(fig)


def compute_angles_between_rfs(model_type: str, dataset: str = "mnist", 
                              compare_final_epoch: bool = False,
                              num_models: int = 10, num_epochs: int = 60) -> None:
    """
    Compute the angles between the receptive fields of all neurons over all epochs for all models.

    Args:
        model_type (str): Type of model to compute angles for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        compare_final_epoch (bool): If True, compare each epoch with the final epoch; if False, compare consecutive epochs.
        num_models (int): Number of models to compute angles for.
        num_epochs (int): Number of epochs to compute angles for.

    Returns:
        None: Saves the angles between RFs to a file.
    """
    # Set dataset-specific parameters
    if dataset.lower() == "mnist":
        MAX_NEURONS = 32
    elif dataset.lower() == "cifar":
        MAX_NEURONS = 128
    
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
    angles_file = f"Results/{dataset}_{model_type}_rf_stability_{comparison_type}_angles.npy"

    if os.path.exists(angles_file):
        print(f"Loading existing angle calculations from {angles_file}")
        return None

    rf_matrices = np.load(f"Results/{dataset}_{model_type}_rfs.npy", allow_pickle=True)
    
    if compare_final_epoch:
        # Compare each epoch with the final epoch
        angles_matrix = np.zeros((num_models, MAX_NEURONS, num_epochs-1))
        
        for model in range(num_models):
            for neuron in range(MAX_NEURONS):
                final_epoch_rf = rf_matrices[model][-1][neuron]
                
                for epoch in range(num_epochs-1):
                    angle = cosine_angle_between_vectors(
                        rf_matrices[model][epoch][neuron], 
                        final_epoch_rf
                    )
                    angles_matrix[model, neuron, epoch] = angle
    else:
        # Compare consecutive epochs
        angles_matrix = np.zeros((num_models, MAX_NEURONS, num_epochs-1))

        for model in range(num_models):
            for epoch in range(num_epochs-1):
                for neuron in range(MAX_NEURONS):
                    # Skip if neuron index is out of range for this epoch
                    if neuron >= len(rf_matrices[model][epoch]) or neuron >= len(rf_matrices[model][epoch + 1]):
                        angles_matrix[model, neuron, epoch] = np.nan
                        continue
                        
                    angle = cosine_angle_between_vectors(
                        rf_matrices[model][epoch][neuron], 
                        rf_matrices[model][epoch + 1][neuron]
                    )
                    angles_matrix[model, neuron, epoch] = angle

    angles_matrix = np.array(angles_matrix)
    np.save(angles_file, angles_matrix)


def compute_average_angles_matrix(model_type: str, dataset: str = "mnist", 
                                 compare_final_epoch: bool = False) -> tuple:
    """
    Compute the average angles between RFs over all models and epochs.

    Args:
        model_type (str): Type of model to compute angles for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        compare_final_epoch (bool): If True, compare each epoch with the final epoch; if False, compare consecutive epochs.
    
    Returns:
        average_angles_matrix (np.ndarray): Matrix of average angles between RFs over all models and epochs.
        non_computable_cells (np.ndarray): Matrix of non-computable cells in the average angles matrix.
    """
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
    angles_matrix = np.load(f"Results/{dataset}_{model_type}_rf_stability_{comparison_type}_angles.npy")
    average_angles_matrix = np.mean(angles_matrix, axis=0)

    if model_type == "sae":
        return average_angles_matrix, None
    elif model_type == "dae":
        highlighted_non_computable_angles = average_angles_matrix.copy()

        for i in range(highlighted_non_computable_angles.shape[0]):
            for j in range(highlighted_non_computable_angles.shape[1] - 1):
                if np.isnan(highlighted_non_computable_angles[i, j]) and not np.isnan(highlighted_non_computable_angles[i, j + 1]):
                    highlighted_non_computable_angles[i, j] = 100

        mask = highlighted_non_computable_angles == 100
        non_computable_cells = np.where(mask, 1, np.nan)

    return average_angles_matrix, non_computable_cells


def create_heatmap(model_type: str, dataset: str = "mnist", 
                  compare_final_epoch: bool = False) -> None:
    """
    Create a heatmap of the average angles between RFs over all models and epochs.
    
    Args:
        model_type (str): Type of model to create heatmap for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        compare_final_epoch (bool): If True, compare each epoch with the final epoch; if False, compare consecutive epochs.
        
    Returns:
        None: Saves the heatmap to a file.
    """
    if dataset.lower() == "mnist":
        fig_size = (12, 7)
    elif dataset.lower() == "cifar":
        fig_size = (10, 10)
    
    angle_matrix, non_computable_cells = compute_average_angles_matrix(model_type, dataset, compare_final_epoch)
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    heatmap = sns.heatmap(
        angle_matrix[:, :],
        cmap="viridis",
        vmin=0,
        vmax=90,
        cbar_kws={"label": "Angle between RFs"},
        square=True
    )

    if model_type == "dae":
        cmap_grey = ListedColormap(['grey'])
        sns.heatmap(
            non_computable_cells[:, :],
            cmap=cmap_grey,
            cbar=False,
            alpha=1,
            square=True
        )

    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0°", "45°", "90°"], fontsize=24)
    cbar.set_label("Receptive Field Angle Difference", fontsize=24)
    cbar.minorticks_off()

    max_epochs = angle_matrix.shape[1]
    mid_epoch = max_epochs // 2
    
    if compare_final_epoch:
        ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
        ax.set_xticklabels(["1", f"{mid_epoch}", f"{max_epochs-1}"], 
                           fontsize=24, rotation=0)
    else:
        ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
        ax.set_xticklabels(["1-2", f"{mid_epoch}-{mid_epoch+1}", f"{max_epochs}-{max_epochs+1}"], 
                           fontsize=24, rotation=0)

    num_pcs = angle_matrix.shape[0]
    mid_pc = num_pcs // 2
    ax.set_yticks([0.5, mid_pc - 0.5, num_pcs - 0.5])
    ax.set_yticklabels(["1", f"{mid_pc}", f"{num_pcs}"], fontsize=24, rotation=90)
    
    if model_type == "sae":
        ax.set_title(f"Stability of Receptive Fields (AE, {dataset.upper()})", fontsize=28, pad=25)
    elif model_type == "dae":
        ax.set_title(f"Stability of Receptive Fields (DevAE, {dataset.upper()})", fontsize=28, pad=25)
    
    ax.set_xlabel("Epoch Comparison", fontsize=24)
    ax.set_ylabel("Neuron Index", fontsize=24)

    comparison_type = "vs_final" if compare_final_epoch else "consecutive"
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_stability_of_rfs_{comparison_type}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_{model_type}_stability_of_rfs_{comparison_type}.svg", bbox_inches="tight")
    plt.close(fig)


def analyze_rf_stability(model_type: str, dataset: str = "mnist", 
                        compare_final_epoch: bool = False,
                        size_ls: list = None, num_models: int = 10, num_epochs: int = 60) -> None:
    """
    Analyze the stability of receptive fields over all models and epochs for a specific model type.
    
    Args:
        model_type (str): Type of model to analyze RF stability for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        compare_final_epoch (bool): If True, compare each epoch with the final epoch; if False, compare consecutive epochs.
        size_ls (list): List of sizes for each epoch for DAE models.
        num_models (int): Number of models to analyze RF stability for.
        num_epochs (int): Number of epochs to analyze RF stability for.
    
    Returns:
        None: Computes and saves RF stability results.
    """
    comparison_type = "comparing to final epoch" if compare_final_epoch else "comparing consecutive epochs"
    print(f"Starting RF stability analysis for {model_type} on {dataset} ({comparison_type})")
    
    compute_angles_between_rfs(model_type, dataset, compare_final_epoch, num_models, num_epochs)
    create_heatmap(model_type, dataset, compare_final_epoch)
    plot_rfs_for_single_model(model_type, dataset, model_idx=0, epoch_idx=-1)
    plot_rf_over_time(model_type, dataset, model_idx=0, neuron_idx=0)
    
    print(f"RF stability analysis for {model_type} on {dataset} complete.")