import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autoencoder import *
from model_utils import *
from solver import *


def load_hidden_layer_statistics(model_type: str, model_arch: str) -> tuple:
    dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'
    hidden_layer_results_path = f"Results/hidden_layer_act_{model_type}_{dataset}.npy"
    neuron_activations = np.load(hidden_layer_results_path)

    # Get the mean activations (index 0) for the specified epoch
    # neuron_activations shape: (num_models, num_epochs, num_layers, 4 statistics values, neurons_per_layer)
    epoch = 0
    mean_activations = neuron_activations[:, :, 0, :]
    model_averaged = np.nanmean(mean_activations, axis=0)
    layer_averages = np.nanmean(model_averaged, axis=1)

    # Get the standard deviation of activations (index 1) for the specified epoch
    mean_activations_std = neuron_activations[:, :, 1, :]
    model_averaged_std = np.nanmean(mean_activations_std, axis=0)
    layer_std = np.nanmean(model_averaged_std, axis=1)

    return layer_averages, layer_std


def plot_neuron_activations(model_arch: str) -> None:
    """
    Plot activations given a specific model architecture with two subplots:
    left for SAE statistics and right for DAE statistics.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        model_arch: Architecture type ('nonlinear' or 'conv')
        
    Returns:
        None: Saves plot to file
    """    
    dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'

    sae_layer_averages, sae_layer_std = load_hidden_layer_statistics('sae', model_arch)
    dae_layer_averages, dae_layer_std = load_hidden_layer_statistics('dae', model_arch)

    if model_arch == 'conv':
        sae_layer_averages = sae_layer_averages[:-1]
        sae_layer_std = sae_layer_std[:-1]
        dae_layer_averages = dae_layer_averages[:-1]
        dae_layer_std = dae_layer_std[:-1]
    
    plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    if model_arch == 'nonlinear':
        x_labels = ['Enc 1', 'Enc 2', 'Bottleneck', 'Dec 1', 'Dec 2']
    elif model_arch == 'conv':
        x_labels = [
            'Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5',
            'Linear', 'L-Out', 'DeConv1', 'Conv6',
            'DeConv2', 'Conv7'
        ]

    # Plot SAE
    x_indices = np.arange(len(sae_layer_averages))
    sae_bars = ax1.bar(x_indices, sae_layer_averages, color='#1a7adb', yerr=sae_layer_std, capsize=5)
    ax1.set_xticks(x_indices)
    if model_arch == 'conv':
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Activation')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    x_indices = np.arange(len(dae_layer_averages))
    dae_bars = ax2.bar(x_indices, dae_layer_averages, color='#e82817', yerr=dae_layer_std, capsize=5)
    ax2.set_xticks(x_indices)
    if model_arch == 'conv':
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax2.set_xticklabels(x_labels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')


    ax2.legend([sae_bars, dae_bars], ['AE', 'DevAE'], loc='upper right')

    max_val = max(
        max(sae_layer_averages) + max(sae_layer_std),
        max(dae_layer_averages) + max(dae_layer_std)
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    fig.suptitle(f'Hidden Layer Activations ({dataset.upper()})', fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    

    plt.savefig(f"Results/hidden_layer_activations_{dataset}.png")
    plt.close()
    
    return None


# def load_zero_activation_statistics(model_type: str, model_arch: str) -> tuple:
#     """
#     Load zero activation statistics for the specified model type and architecture.
    
#     Args:
#         model_type: Type of model ('sae' or 'dae')
#         model_arch: Architecture type ('nonlinear' or 'conv')
        
#     Returns:
#         tuple: Statistics about zero activations categories
#     """
#     dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'
#     hidden_layer_results_path = f"Results/hidden_layer_act_{model_type}_{dataset}.npy"
#     neuron_activations = np.load(hidden_layer_results_path)
    
#     # Get the always zero percentages (index 4)
#     always_zero_activations = neuron_activations[:, :, 4, :]
#     model_averaged_always = np.nanmean(always_zero_activations, axis=0)
#     always_zero_averages = np.nanmean(model_averaged_always, axis=1)
    
#     # Get the never zero percentages (index 5)
#     never_zero_activations = neuron_activations[:, :, 5, :]
#     model_averaged_never = np.nanmean(never_zero_activations, axis=0)
#     never_zero_averages = np.nanmean(model_averaged_never, axis=1)
    
#     # Get the sometimes zero percentages (index 6)
#     sometimes_zero_activations = neuron_activations[:, :, 6, :]
#     model_averaged_sometimes = np.nanmean(sometimes_zero_activations, axis=0)
#     sometimes_zero_averages = np.nanmean(model_averaged_sometimes, axis=1)
    
#     return always_zero_averages, never_zero_averages, sometimes_zero_averages


# def plot_zero_activations(model_arch: str) -> None:
#     """
#     Plot zero activations as a stacked bar chart showing always zero, never zero and sometimes zero percentages.
    
#     Args:
#         model_arch: Architecture type ('nonlinear' or 'conv')
        
#     Returns:
#         None: Saves plot to file
#     """
#     dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'
    
#     # Load statistics for SAE
#     sae_always_zero, sae_never_zero, sae_sometimes_zero = load_zero_activation_statistics('sae', model_arch)
    
#     # Load statistics for DAE
#     dae_always_zero, dae_never_zero, dae_sometimes_zero = load_zero_activation_statistics('dae', model_arch)
    
#     # Remove the last layer if needed
#     sae_always_zero = sae_always_zero[:-1]
#     sae_never_zero = sae_never_zero[:-1]
#     sae_sometimes_zero = sae_sometimes_zero[:-1]
    
#     dae_always_zero = dae_always_zero[:-1]
#     dae_never_zero = dae_never_zero[:-1]
#     dae_sometimes_zero = dae_sometimes_zero[:-1]
    
#     plt.rc('font', size=20)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
#     # Set x-axis labels based on architecture
#     if model_arch == 'nonlinear':
#         x_labels = ['Enc 1', 'Enc 2', 'Bottleneck', 'Dec 1', 'Dec 2']
#     elif model_arch == 'conv':
#         x_labels = [
#             'Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5',
#             'Linear', 'L-Out', 'DeConv1', 'Conv6',
#             'DeConv2', 'Conv7'
#         ]
    
#     # Plot SAE data on the left subplot as stacked bar
#     x_indices = np.arange(len(sae_always_zero))
    
#     # Create stacked bar for SAE
#     sae_bottom_sometimes = sae_always_zero
#     sae_bottom_never = sae_always_zero + sae_sometimes_zero
    
#     ax1.bar(x_indices, sae_always_zero, color='#ff6961', label='Always Zero (Dead)')
#     ax1.bar(x_indices, sae_sometimes_zero, bottom=sae_bottom_sometimes, color='#77dd77', label='Sometimes Zero')
#     ax1.bar(x_indices, sae_never_zero, bottom=sae_bottom_never, color='#1a7adb', label='Never Zero')
    
#     ax1.set_xticks(x_indices)
#     if model_arch == 'conv':
#         ax1.set_xticklabels(x_labels, rotation=45, ha='right')
#     else:
#         ax1.set_xticklabels(x_labels)
#     ax1.set_ylabel('Percentage of Neurons')
#     ax1.set_ylim(0, 100)  # Percentage scale
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.set_title('AE')
#     ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
#     # Plot DAE data on the right subplot as stacked bar
#     x_indices = np.arange(len(dae_always_zero))
    
#     # Create stacked bar for DAE
#     dae_bottom_sometimes = dae_always_zero
#     dae_bottom_never = dae_always_zero + dae_sometimes_zero
    
#     ax2.bar(x_indices, dae_always_zero, color='#ff6961', label='Always Zero (Dead)')
#     ax2.bar(x_indices, dae_sometimes_zero, bottom=dae_bottom_sometimes, color='#77dd77', label='Sometimes Zero')
#     ax2.bar(x_indices, dae_never_zero, bottom=dae_bottom_never, color='#1a7adb', label='Never Zero')
    
#     ax2.set_xticks(x_indices)
#     if model_arch == 'conv':
#         ax2.set_xticklabels(x_labels, rotation=45, ha='right')
#     else:
#         ax2.set_xticklabels(x_labels)
#     ax2.set_ylim(0, 100)  # Percentage scale
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
#     ax2.yaxis.set_visible(False)
#     ax2.set_ylabel('')
#     ax2.set_title('DevAE')
    
#     fig.suptitle(f'Neuronal Activation Patterns - {dataset.upper()}', fontsize=24)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjusted bottom to make room for legend
    
#     plt.savefig(f"Results/neuron_activation_patterns_{dataset}.png")
#     plt.close()
    
#     return None


def load_zero_activation_statistics(model_type: str, model_arch: str) -> tuple:
    """
    Load zero activation statistics for the specified model type and architecture.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        model_arch: Architecture type ('nonlinear' or 'conv')
        
    Returns:
        tuple: Statistics about zero activations categories
    """
    dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'
    hidden_layer_results_path = f"Results/hidden_layer_act_{model_type}_{dataset}.npy"
    neuron_activations = np.load(hidden_layer_results_path)
    
    always_zero_activations = neuron_activations[:, :, 4, :]
    model_averaged_always = np.nanmean(always_zero_activations, axis=0)
    always_zero_averages = np.nanmean(model_averaged_always, axis=1)
    
    never_zero_activations = neuron_activations[:, :, 5, :]
    model_averaged_never = np.nanmean(never_zero_activations, axis=0)
    never_zero_averages = np.nanmean(model_averaged_never, axis=1)
    
    sometimes_zero_activations = neuron_activations[:, :, 6, :]
    model_averaged_sometimes = np.nanmean(sometimes_zero_activations, axis=0)
    sometimes_zero_averages = np.nanmean(model_averaged_sometimes, axis=1)
    
    return always_zero_averages, never_zero_averages, sometimes_zero_averages


def plot_zero_activations(model_arch: str) -> None:
    """
    Plot zero activations as a stacked bar chart showing always zero, never zero and sometimes zero percentages.
    
    Args:
        model_arch: Architecture type ('nonlinear' or 'conv')
        
    Returns:
        None: Saves plot to file
    """
    dataset = 'mnist' if model_arch == 'nonlinear' else 'cifar'
    
    sae_always_zero, sae_never_zero, sae_sometimes_zero = load_zero_activation_statistics('sae', model_arch)
    dae_always_zero, dae_never_zero, dae_sometimes_zero = load_zero_activation_statistics('dae', model_arch)
    
    if model_arch == 'conv':
        sae_always_zero = sae_always_zero[:-1]
        sae_never_zero = sae_never_zero[:-1]
        sae_sometimes_zero = sae_sometimes_zero[:-1]
        
        dae_always_zero = dae_always_zero[:-1]
        dae_never_zero = dae_never_zero[:-1]
        dae_sometimes_zero = dae_sometimes_zero[:-1]
    
    plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    if model_arch == 'nonlinear':
        x_labels = ['Enc 1', 'Enc 2', 'Bottleneck', 'Dec 1', 'Dec 2']
    elif model_arch == 'conv':
        x_labels = [
            'Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5',
            'Linear', 'L-Out', 'DeConv1', 'Conv6',
            'DeConv2', 'Conv7'
        ]
    
    # Plot SAE
    x_indices = np.arange(len(sae_always_zero))
    
    sae_bottom_sometimes = sae_always_zero
    sae_bottom_never = sae_always_zero + sae_sometimes_zero
    
    bar1 = ax1.bar(x_indices, sae_always_zero, color='#D72638', label='Always Zero (Dead)')
    bar2 = ax1.bar(x_indices, sae_sometimes_zero, bottom=sae_bottom_sometimes, color='#e8b81c', label='Sometimes Zero')
    bar3 = ax1.bar(x_indices, sae_never_zero, bottom=sae_bottom_never, color='#1B998B', label='Never Zero')
    
    ax1.set_xticks(x_indices)
    if model_arch == 'conv':
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Percentage of Neurons')
    ax1.set_ylim(0, 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('AE')
    
    # Plot DAE
    x_indices = np.arange(len(dae_always_zero))
    
    dae_bottom_sometimes = dae_always_zero
    dae_bottom_never = dae_always_zero + dae_sometimes_zero
    
    ax2.bar(x_indices, dae_always_zero, color='#D72638', label='Always Zero (Dead)')
    ax2.bar(x_indices, dae_sometimes_zero, bottom=dae_bottom_sometimes, color='#e8b81c', label='Sometimes Zero')
    ax2.bar(x_indices, dae_never_zero, bottom=dae_bottom_never, color='#1B998B', label='Never Zero')
    
    ax2.set_xticks(x_indices)
    if model_arch == 'conv':
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, 100)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    ax2.set_title('DevAE')

    fig.legend([bar1, bar2, bar3], 
              ['Inactive\nNeurons', 'Conditionally\nActive\nNeurons', 'Universally\nActive\nNeurons'], 
              loc='center', 
              bbox_to_anchor=(0.525, 0.6),
              frameon=True,
              fontsize=16,
              labelspacing=1.5)
    
    fig.suptitle(f'Neuron Activation Sparsity Across Network Layers ({dataset.upper()})', fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)
    
    plt.savefig(f"Results/neuron_activation_patterns_{dataset}.png")
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


def plot_hidden_layer_activation(model_arch: str) -> None:
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
    plot_neuron_activations(model_arch=model_arch)
    plot_zero_activations(model_arch=model_arch)
    return None