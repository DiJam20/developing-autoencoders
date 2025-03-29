import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bottleneck_activation import compute_bottleneck_activation

DATASET = 'cifar'
NUM_MODELS = 40
BASE_PATH = "/home/david/"

LABEL_SIZE = 22
TICK_SIZE = 22
LEGEND_SIZE = 22
TITLE_SIZE = 26

def create_figure_4(dataset='cifar'):
    """
    Create Figure 4 with two subplots:
    A: Mean activation per neuron group (left)
    B: Percentage of zero activations per neuron group (right)
    
    Args:
        dataset: Dataset name ('cifar')
    """    
    fig = plt.figure(figsize=(20, 6))
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    ax_left = plt.subplot(gs[0])
    ax_right = plt.subplot(gs[1])
    
    plot_zeros_per_neuron(ax_left, dataset)
    plot_activation_per_neuron(ax_right, dataset)
    
    # Add figure labels
    label_x, label_y = -0.15, 1.05
    ax_left.text(label_x, label_y, 'A', transform=ax_left.transAxes, 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax_right.text(label_x, label_y, 'B', transform=ax_right.transAxes, 
                 fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"Results/figures/png/{dataset}_figure_4.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_figure_4.svg", bbox_inches='tight')
    plt.savefig(f"Results/figures/eps/{dataset}_figure_4.eps", bbox_inches='tight')
    plt.close()


def plot_activation_per_neuron(ax, dataset="cifar"):
    """
    Plot the mean activation per neuron for the SAE and DAE models,
    grouped by neuron ranges and displayed as a bar chart with error bars.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [4, 10, 16, 24, 32]
    elif dataset.lower() == "cifar":
        neuron_groups = [6, 10, 16, 28, 48, 90, 128]
    
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    
    results = np.load(result_file, allow_pickle=True).item()
    sae_mean = abs(results['mean_sae'].squeeze())
    dae_mean = abs(results['mean_dae'].squeeze())

    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean and std for each group
    sae_group_means = []
    dae_group_means = []
    sae_group_stds = []
    dae_group_stds = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = sae_mean[start:end]
        dae_group = dae_mean[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        dae_group_stds.append(np.std(dae_group))
    
    x_indices = np.arange(len(neuron_groups))
    width = 0.35
    
    ax.bar(x_indices - width/2, sae_group_means, width, label='AE', color='#1a7adb',
          yerr=sae_group_stds, capsize=5)
    ax.bar(x_indices + width/2, dae_group_means, width, label='Dev-AE', color='#e82817',
          yerr=dae_group_stds, capsize=5)
    
    ax.set_xlabel('Neuron Groups', fontsize=LABEL_SIZE)
    ax.set_ylabel('Mean Activation', fontsize=LABEL_SIZE)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=90)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE)


def plot_zeros_per_neuron(ax, dataset="cifar"):
    """
    Plot the percentage of zero activations per neuron for the SAE and DAE models,
    grouped by neuron ranges and displayed as a grouped bar chart.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [4, 10, 16, 24, 32]
    elif dataset.lower() == "cifar":
        neuron_groups = [6, 10, 16, 28, 48, 90, 128]
    
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros'].squeeze()
    mean_dae_zeros = results['mean_dae_zeros'].squeeze()
    
    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean for each group
    sae_group_means = []
    dae_group_means = []
    sae_group_stds = []
    dae_group_stds = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = mean_sae_zeros[start:end]
        dae_group = mean_dae_zeros[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        dae_group_stds.append(np.std(dae_group))
    
    x_indices = np.arange(len(neuron_groups))
    width = 0.35
    
    # Scale up the percentages by 1000
    scale_factor = 1000
    
    # Scale the data
    sae_group_means_scaled = [x * scale_factor for x in sae_group_means]
    dae_group_means_scaled = [x * scale_factor for x in dae_group_means]
    sae_group_stds_scaled = [x * scale_factor for x in sae_group_stds]
    dae_group_stds_scaled = [x * scale_factor for x in dae_group_stds]
    
    ax.bar(x_indices - width/2, sae_group_means_scaled, width, label='AE', color='#1a7adb',
          yerr=sae_group_stds_scaled, capsize=5)
    ax.bar(x_indices + width/2, dae_group_means_scaled, width, label='Dev-AE', color='#e82817',
          yerr=dae_group_stds_scaled, capsize=5)
    
    ax.set_xlabel('Neuron Groups', fontsize=LABEL_SIZE)
    ax.set_ylabel('% Zero Activation (×10³)', fontsize=LABEL_SIZE)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=90)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    # ax.legend(loc='upper right', fontsize=LEGEND_SIZE)


compute_bottleneck_activation(NUM_MODELS, DATASET, BASE_PATH)
create_figure_4(DATASET)