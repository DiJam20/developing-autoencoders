import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from autoencoder import *
from solver import *
from model_utils import *


IMG_WIDTH, IMG_HEIGHT = 28, 28
MAX_NEURONS = 32


def display_ready_rf(rf: np.ndarray) -> np.ndarray:
    """
    Reshape the RF to be displayed as an image.

    Args:
        rf (np.ndarray): Receptive field to reshape.
    
    Returns:
        np.ndarray: Reshaped receptive field.
    """
    if len(rf.shape) == 1:
        return rf.reshape(IMG_WIDTH, IMG_HEIGHT)
    return rf


def plot_rfs_for_single_model(model_type: str, model_idx: int = 0, epoch_idx: int = -1) -> None:
    """
    Plot receptive fields for all neurons in a model at a specific epoch.
    
    Args:
        model_type (str): Type of model ('sae' or 'dae')
        model_idx (int): Index of the model to visualize
        epoch_idx (int): Index of epoch to visualize (-1 for last epoch)

    Returns:
        None: Plots the receptive fields and saves the figure.
    """
    rf_matrices = np.load(f"Results/{model_type}_rfs.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
    rf_ls = rf_matrix[epoch_idx]
    
    fig = plt.figure(figsize=(20,9))
    fig.suptitle(f"Receptive Fields ({model_type})", fontsize=24)
    
    for i in range(MAX_NEURONS):
        # MATCH SUBPLOT VALUES TO NUMBER OF NEURONS
        plt.subplot(4,8,i+1)
        plt.title(str(i+1))
        plt.axis('off')
        
        # Only plot if the index exists in rf_ls, otherwise leave empty
        if i < len(rf_ls):
            plt.imshow(display_ready_rf(rf_ls[i]), cmap='gray')
        else:
            plt.imshow(np.zeros((IMG_WIDTH, IMG_HEIGHT)), cmap='gray')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    epoch_label = epoch_idx if epoch_idx >= 0 else f"final"
    plt.savefig(f"Results/{model_type}_rfs_model_{model_idx}_epoch_{epoch_label}.png", dpi=300)
    plt.savefig(f"Results/{model_type}_rfs_model_{model_idx}_epoch_{epoch_label}.svg")
    plt.close(fig)


def plot_rf_over_time(model_type: str, model_idx: int, neuron_idx: int) -> None:
    """
    Plot the receptive field development of a single neuron over all epochs for a single model.

    Args:
        model_type (str): Type of model to plot RF development for.
        model_idx (int): Index of the model to plot RF development for.
        neuron_idx (int): Index of the neuron to plot RF development for.

    Returns:
        None: Plots the receptive field development and saves the figure.
    """
    rf_matrices = np.load(f"Results/{model_type}_rfs.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
        
    # Calculate grid dimensions
    cols = 5
    rows = (len(rf_matrix) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))
    axes = axes.ravel()
    
    for i in range(len(rf_matrix)):
        axes[i].imshow(display_ready_rf(rf_matrix[i][neuron_idx]), cmap='gray')
        axes[i].axis('off')

    fig.suptitle(f"Receptive Field Development of Neuron 0 ({model_type})", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f"Results/{model_type}_rf_development_model_{model_idx}_neuron_{neuron_idx}.png", dpi=300)
    plt.savefig(f"Results/{model_type}_rf_development_model_{model_idx}_neuron_{neuron_idx}.svg")
    plt.close(fig)


def compute_angles_between_rfs(model_type: str, num_models: int, num_epochs: int, num_neurons: int) -> None:
    """
    Compute the angles between the receptive fields of all neurons over all epochs for all models.

    Args:
        model_type (str): Type of model to compute angles for.
        num_models (int): Number of models to compute angles for.
        num_epochs (int): Number of epochs to compute angles for.
        num_neurons (int): Number of neurons in the bottleneck layer.

    Returns:
        None: Saves the angles between RFs to a file.
    """
    angles_file = f"Results/{model_type}_rf_stability_angles.npy"

    if os.path.exists(angles_file):
        print(f"Loading existing angle calculations from {angles_file}")
        return np.load(angles_file)

    rf_matrices = np.load(f"Results/{model_type}_rfs.npy", allow_pickle=True)
    
    angles_matrix = np.zeros((num_models, num_neurons, num_epochs-1))

    for model in range(num_models):
        for epoch in range(num_epochs-1):
            for neuron in range(num_neurons):
                angle = cosine_angle_between_pcs(rf_matrices[model][epoch][neuron], rf_matrices[model][epoch + 1][neuron])
                angles_matrix[model, neuron, epoch] = angle

    angles_matrix = np.array(angles_matrix)

    np.save(angles_file, angles_matrix)


def compute_average_angles_matrix(model_type: str) -> tuple:
    """
    Compute the average angles between RFs over all models and epochs.

    Args:
        model_type (str): Type of model to compute angles for.
    
    Returns:
        average_angles_matrix (np.ndarray): Matrix of average angles between RFs over all models and epochs.
        non_computable_cells (np.ndarray): Matrix of non-computable cells in the average angles matrix.
    """
    angles_matrix = np.load(f"Results/{model_type}_rf_stability_angles.npy")
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


def create_heatmap(model_type: str) -> None:
    """
    Create a heatmap of the average angles between RFs over all models and epochs.
    
    Args:
        model_type (str): Type of model to create heatmap for.
        
    Returns:
        None: Saves the heatmap to a file.
    """
    angle_matrix, non_computable_cells = compute_average_angles_matrix(model_type)
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    heatmap = sns.heatmap(
        angle_matrix[:, :],
        cmap="viridis",
        vmin=0,
        vmax=90,
        cbar_kws={"label": "Angle between PCs"},
        linewidths=0.5,
        square=True,
    )

    if model_type == "dae":
        cmap_grey = ListedColormap(['grey'])
        sns.heatmap(
            non_computable_cells[:, :],
            cmap=cmap_grey,
            cbar=False,
            alpha=1,
            linewidths=0.5,
            square=True,
        )

    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0°", "45°", "90°"], fontsize=24)
    cbar.set_label("Receptive Field Angle Difference", fontsize=24)
    cbar.minorticks_off()

    max_epochs = angle_matrix.shape[1]
    mid_epoch = max_epochs // 2
    ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
    ax.set_xticklabels(["1-2", f"{mid_epoch}-{mid_epoch+1}", f"{max_epochs}-{max_epochs+1}"], 
                        fontsize=24, rotation=0)

    num_pcs = angle_matrix.shape[0]
    mid_pc = num_pcs // 2
    ax.set_yticks([0.5, mid_pc - 0.5, num_pcs - 0.5])
    ax.set_yticklabels(["1", f"{mid_pc}", f"{num_pcs}"], fontsize=24, rotation=90)

    if model_type == "sae":
        ax.set_title("Stability of Receptive Fields (AE)", fontsize=28, pad=25)
    elif model_type == "dae":
        ax.set_title("Stability of Receptive Fields (Dev-AE)", fontsize=28, pad=25)
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Neuron Index", fontsize=24)

    plt.savefig(f"Results/{model_type}_stability_of_rfs.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"Results/{model_type}_stability_of_rfs.svg", bbox_inches="tight")
    plt.close(fig)


def analyze_rf_stability(model_type: str, size_ls: list = None, num_models: int = 10, num_epochs: int = 60) -> None:
    """
    Analyze the stability of receptive fields over all models and epochs for a specific model type.
    
    Args:
        model_type (str): Type of model to analyze RF stability for.
        size_ls (list): List of sizes for each epoch for DAE models.
        num_models (int): Number of models to analyze RF stability for.
        num_epochs (int): Number of epochs to analyze RF stability for.
    
    Returns:
        None: Computes and saves RF stability results.
    """
    compute_angles_between_rfs(model_type, num_models, num_epochs, MAX_NEURONS)
    create_heatmap(model_type)
    plot_rfs_for_single_model(model_type, model_idx=0, epoch_idx=-1)
    plot_rf_over_time(model_type, model_idx=0, neuron_idx=0)