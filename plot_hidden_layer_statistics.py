import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autoencoder import *
from model_utils import *
from solver import *


def plot_neuron_activations(model_type: str, epoch: int = 59) -> None:
    """
    Plot activations given a specific RF and epoch (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        epoch: Epoch to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    
    # Get the mean activations (index 0) for the specified epoch
    # neuron_activations shape: (num_models, num_epochs, num_layers, 4 statistics values, neurons_per_layer)
    mean_activations = neuron_activations[:, epoch, :, 0, :]
    model_averaged = np.nanmean(mean_activations, axis=0)
    layer_averages = np.nanmean(model_averaged, axis=1)

    # Get the standard deviation of activations (index 1) for the specified epoch
    mean_activations_std = neuron_activations[:, epoch, :, 1, :]
    model_averaged_std = np.nanmean(mean_activations_std, axis=0)
    layer_std = np.nanmean(model_averaged_std, axis=1)

    plt.rc('font', size=16)

    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(layer_averages, label='AE', color='blue', linewidth=4)
    plt.fill_between(range(len(layer_averages)), 
                    layer_averages - layer_std, 
                    layer_averages + layer_std,
                    color='blue', alpha=0.1)

    if model_type == 'sae':
        model_type = 'AE'
    if model_type == 'dae':
        model_type = 'Dev-AE'
    plt.title(f'Neuron Activation Across Hidden Layers ({model_type})')
    plt.ylabel('Mean Activation Value')
    plt.xticks(range(5), ['Encoder [512]', 'Encoder [128]', 'Bottleneck [32]', 'Decoder [128]', 'Decoder [512]'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_hidden_layer_neuron_activations_epoch_{epoch}.png")
    plt.close()

    return None


def plot_zero_activations(model_type: str, epoch: int = 59) -> None:
    """
    Plot zero activations given a specific RF and epoch (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        epoch: Epoch to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")

    # neuron_activations shape: (num_models, num_epochs, num_layers, 4 statistics values, neurons_per_layer)
    mean_activations = neuron_activations[:, epoch, :, 2, :]
    model_averaged = np.nanmean(mean_activations, axis=0)
    zero_averages = np.nanmean(model_averaged, axis=1)

    mean_activations_std = neuron_activations[:, epoch, :, 3, :]
    model_averaged_std = np.nanmean(mean_activations_std, axis=0)
    zero_std = np.nanmean(model_averaged_std, axis=1)

    plt.rc('font', size=16)

    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(zero_averages, label='AE', color='blue', linewidth=4)
    plt.fill_between(range(len(zero_averages)), 
                    zero_averages - zero_std, 
                    zero_averages + zero_std,
                    color='blue', alpha=0.1)

    if model_type == 'sae':
        model_type = 'AE'
    if model_type == 'dae':
        model_type = 'Dev-AE'
    plt.title('Percentage of 0s Across Hidden Layers')
    plt.ylabel('Percentage of 0s')
    plt.ylim(0, 100)
    plt.xticks(range(5), ['Encoder [512]', 'Encoder [128]', 'Bottleneck [32]', 'Decoder [128]', 'Decoder [512]'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_hidden_layer_zero_activations_epoch_{epoch}.png")
    plt.close()

    return None


def activations_heatmap(model_type: str, layer_idx: int = 2) -> None:
    """
    Generate a heatmap of neuron activations over epochs.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        layer_idx: Index of the layer to visualize (default: 2, which is the bottleneck layer)

    Returns:
        None: Saves plot to file
    """
    neurons_per_layer = [512, 128, 32, 128, 512]

    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    epoch_activations_mean = np.nanmean(neuron_activations[:, :, layer_idx, 0, :], axis=0)

    num_neurons = neurons_per_layer[layer_idx]
    epoch_activations_mean = epoch_activations_mean[:, :num_neurons]

    epoch_activations_mean = epoch_activations_mean.T
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    num_neurons = epoch_activations_mean.shape[0]
    
    heatmap = sns.heatmap(
        epoch_activations_mean,
        cmap="inferno",
        vmin=0,
        vmax=10,
        cbar_kws={"label": "Angle between PCs"},
        linewidths=0.5,
        square=True
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Strength of Activation", fontsize=24)
    cbar.minorticks_off()

    # Set x-axis ticks (epochs)
    num_epochs = max(epoch_activations_mean.shape[1], 30)
    ax.set_xticks([0.5, num_epochs//2 - 0.5, num_epochs - 0.5])
    ax.set_xticklabels(["1", str(num_epochs//2), str(num_epochs)], fontsize=24, rotation=0)
    
    # Set y-axis ticks (neurons)
    y_ticks = [0.5, num_neurons//2 - 0.5, num_neurons - 0.5]
    y_labels = ["1", str(num_neurons//2), str(num_neurons)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=24, rotation=90)

    ax.set_title(f"Neuron Activation over Epochs ({model_type})", fontsize=28, pad=25)
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Neuron Index", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_neuron_activations_over_time.png")
    plt.close()
    
    return None


def plot_hidden_layer_activation(model_type: str, epoch: int) -> None:
    """
    Compute RF specificity for all models.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        num_models: Number of models to process
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
    Results:
        None: Saves plots to file
    """
    plot_neuron_activations(model_type, epoch=epoch)
    plot_zero_activations(model_type, epoch=epoch)
    activations_heatmap(model_type)
    return None