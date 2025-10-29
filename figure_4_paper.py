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
    A: Mean activation per neuron group (top)
    B: Percentage of zero activations per neuron group (bottom)
    
    Args:
        dataset: Dataset name ('cifar')
    """    
    fig = plt.figure(figsize=(10, 12))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    ax_top = plt.subplot(gs[0])
    ax_bottom = plt.subplot(gs[1])
    
    plot_zeros_per_neuron(ax_top, dataset)
    plot_activation_per_neuron(ax_bottom, dataset)
    
    # Add figure labels
    label_x, label_y = -0.15, 1.05
    ax_top.text(label_x, label_y, 'A', transform=ax_top.transAxes, 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom.text(label_x, label_y, 'B', transform=ax_bottom.transAxes, 
                 fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"paper_results/figures/png/{dataset}_figure_4.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"paper_results/figures/pdf/{dataset}_figure_4.pdf", bbox_inches='tight')
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
        neuron_groups = [6, 10, 17, 29, 50, 85, 128]
    
    result_file = f"paper_results/{dataset}_bottleneck_activation.npy"
    
    results = np.load(result_file, allow_pickle=True).item()
    sae_mean = abs(results['mean_sae_nonzero'].squeeze())
    pca_mean = abs(results['mean_pca_nonzero'].squeeze())
    dae_mean = abs(results['mean_dae_nonzero'].squeeze())

    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean and std for each group
    sae_group_means = []
    pca_group_means = []
    dae_group_means = []
    sae_group_stds = []
    pca_group_stds = []
    dae_group_stds = []
    sae_group_data = []
    pca_group_data = []
    dae_group_data = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = sae_mean[start:end]
        pca_group = pca_mean[start:end]
        dae_group = dae_mean[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        pca_group_means.append(np.mean(pca_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        pca_group_stds.append(np.std(pca_group))
        dae_group_stds.append(np.std(dae_group))
        sae_group_data.append(sae_group)
        pca_group_data.append(pca_group)
        dae_group_data.append(dae_group)
    
    x_indices = np.arange(len(neuron_groups))
    width = 0.25
    
    ax.bar(x_indices - width, sae_group_means, width, label='AE', color='#1a7adb', zorder=0)
    ax.bar(x_indices, pca_group_means, width, label='PCA-AE', color='#00a65a', zorder=0)
    ax.bar(x_indices + width, dae_group_means, width, label='Dev-AE', color='#e82817', zorder=0)
    
    ax.errorbar(x_indices - width, sae_group_means, yerr=sae_group_stds, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    ax.errorbar(x_indices, pca_group_means, yerr=pca_group_stds, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    ax.errorbar(x_indices + width, dae_group_means, yerr=dae_group_stds, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    
    for i in range(len(neuron_groups)):
        x_jitter_sae = np.random.normal(0, 0.04, len(sae_group_data[i]))
        x_jitter_pca = np.random.normal(0, 0.04, len(pca_group_data[i]))
        x_jitter_dae = np.random.normal(0, 0.04, len(dae_group_data[i]))
        
        ax.scatter(x_indices[i] - width + x_jitter_sae, sae_group_data[i], 
                  color='#1a7adb', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
        ax.scatter(x_indices[i] + x_jitter_pca, pca_group_data[i], 
                  color='#00a65a', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
        ax.scatter(x_indices[i] + width + x_jitter_dae, dae_group_data[i], 
                  color='#e82817', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Neuron Groups', fontsize=LABEL_SIZE)
    ax.set_ylabel('Mean Activation', fontsize=LABEL_SIZE)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=45)
    ax.tick_params(axis='both', labelsize=TICK_SIZE, width=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
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
        neuron_groups = [6, 10, 17, 29, 50, 85, 128]
    
    result_file = f"paper_results/{dataset}_bottleneck_activation.npy"
    
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros'].squeeze()
    mean_pca_zeros = results['mean_pca_zeros'].squeeze()
    mean_dae_zeros = results['mean_dae_zeros'].squeeze()
    
    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean for each group
    sae_group_means = []
    pca_group_means = []
    dae_group_means = []
    sae_group_stds = []
    pca_group_stds = []
    dae_group_stds = []
    sae_group_data = []
    pca_group_data = []
    dae_group_data = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = mean_sae_zeros[start:end]
        pca_group = mean_pca_zeros[start:end]
        dae_group = mean_dae_zeros[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        pca_group_means.append(np.mean(pca_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        pca_group_stds.append(np.std(pca_group))
        dae_group_stds.append(np.std(dae_group))
        sae_group_data.append(sae_group)
        pca_group_data.append(pca_group)
        dae_group_data.append(dae_group)
    
    x_indices = np.arange(len(neuron_groups))
    width = 0.25
    
    # Scale up the percentages by 1000
    scale_factor = 1
    
    sae_group_means_scaled = [x * scale_factor for x in sae_group_means]
    pca_group_means_scaled = [x * scale_factor for x in pca_group_means]
    dae_group_means_scaled = [x * scale_factor for x in dae_group_means]
    sae_group_stds_scaled = [x * scale_factor for x in sae_group_stds]
    pca_group_stds_scaled = [x * scale_factor for x in pca_group_stds]
    dae_group_stds_scaled = [x * scale_factor for x in dae_group_stds]
    
    ax.bar(x_indices - width, sae_group_means_scaled, width, label='AE', color='#1a7adb', zorder=0)
    ax.bar(x_indices, pca_group_means_scaled, width, label='PCA-AE', color='#00a65a', zorder=0)
    ax.bar(x_indices + width, dae_group_means_scaled, width, label='Dev-AE', color='#e82817', zorder=0)
    
    ax.errorbar(x_indices - width, sae_group_means_scaled, yerr=sae_group_stds_scaled, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    ax.errorbar(x_indices, pca_group_means_scaled, yerr=pca_group_stds_scaled, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    ax.errorbar(x_indices + width, dae_group_means_scaled, yerr=dae_group_stds_scaled, fmt='none', 
                ecolor='black', capsize=5, capthick=2, elinewidth=2, zorder=2)
    
    for i in range(len(neuron_groups)):
        x_jitter_sae = np.random.normal(0, 0.04, len(sae_group_data[i]))
        x_jitter_pca = np.random.normal(0, 0.04, len(pca_group_data[i]))
        x_jitter_dae = np.random.normal(0, 0.04, len(dae_group_data[i]))
        
        ax.scatter(x_indices[i] - width + x_jitter_sae, sae_group_data[i] * scale_factor, 
                  color='#1a7adb', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
        ax.scatter(x_indices[i] + x_jitter_pca, pca_group_data[i] * scale_factor, 
                  color='#00a65a', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
        ax.scatter(x_indices[i] + width + x_jitter_dae, dae_group_data[i] * scale_factor, 
                  color='#e82817', s=20, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
    
    # ax.set_ylabel('% Zero Activation (×10⁻³)', fontsize=LABEL_SIZE)
    ax.set_ylabel('% Zero Activation', fontsize=LABEL_SIZE)


    ax.set_xticks(x_indices)
    ax.set_xticklabels([])
    ax.tick_params(axis='both', labelsize=TICK_SIZE, width=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_ylim(bottom=0)
    # ax.legend(loc='upper right', fontsize=LEGEND_SIZE)


compute_bottleneck_activation(NUM_MODELS, DATASET, BASE_PATH)
create_figure_4(DATASET)