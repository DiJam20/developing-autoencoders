import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats

from importance_for_classification import compute_neuron_importance, get_neuron_group_importance
from frequency_noise import compute_average_frequency_classification
from pc_noise import compute_pc_noise_analysis

DATASET = "cifar"
NUM_MODELS = 40
BASE_PATH = "/home/david/"

LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24
TITLE_SIZE = 26

def create_figure_3(dataset='cifar'):
    """
    Create Figure 3 with three subplots:
    A: PC noise impact rankings (top row)
    B: Classification of frequency noise types (bottom left)
    C: Neuron group importance (bottom right)
    
    Args:
        dataset: Dataset name ('cifar')
    """    
    fig = plt.figure(figsize=(20, 15))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1.2, 1], wspace=0.3, hspace=0.4)
    
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :])
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 0])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1])
    
    ax_top = plt.subplot(gs_top[0])
    ax_bottom_left = plt.subplot(gs_bottom_left[0])
    ax_bottom_right = plt.subplot(gs_bottom_right[0])
    
    plot_pc_rankings(ax_top, dataset)
    plot_frequency_classification(ax_bottom_left, dataset)
    plot_neuron_importance(ax_bottom_right, dataset)
    
    # Add figure labels
    ax_top.text(-0.08, 1.05, 'A', transform=ax_top.transAxes, 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom_left.text(-0.2, 1.1, 'B', transform=ax_bottom_left.transAxes, 
                        fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom_right.text(-0.2, 1.1, 'C', transform=ax_bottom_right.transAxes, 
                         fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"Results/figures/png/{dataset}_figure_3.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_figure_3.svg", bbox_inches='tight')
    plt.close()


def plot_pc_rankings(ax, dataset="cifar"):
    """
    Plot PC noise impact rankings heatmap (adapted from create_ranking_heatmaps)
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('mnist' or 'cifar')
    """
    ax.axis('off')
        
    results = np.load(f"Results/{dataset}_pc_noise.npy", allow_pickle=True).item()
    
    if dataset.lower() == "mnist":
        manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    else:
        manipulated_neurons = [(0, 6), (6, 10), (10, 16), (16, 28), (28, 48), (48, 90), (90, 128)]
    
    sample_data = next(iter(results.values()))[0]
    num_neurons = len(sample_data[0])
    
    sae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))
    dae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))
    
    # Calculate mean activation differences for each PC range and each neuron
    for i, pc_range in enumerate(manipulated_neurons):
        # Get all runs for this PC range
        runs = results[pc_range]
        
        # Stack all SAE and DAE differences for this PC range
        sae_diffs = np.vstack([run[0] for run in runs])
        dae_diffs = np.vstack([run[1] for run in runs])
        
        # Calculate mean across runs
        sae_mean = np.mean(sae_diffs, axis=0)
        dae_mean = np.mean(dae_diffs, axis=0)
        
        # Store in matrices
        sae_activation_matrix[i, :] = sae_mean
        dae_activation_matrix[i, :] = dae_mean
    
    # Calculate rankings for each neuron (1 = highest activation difference)
    sae_rankings = np.zeros_like(sae_activation_matrix, dtype=int)
    dae_rankings = np.zeros_like(dae_activation_matrix, dtype=int)
    
    for neuron in range(num_neurons):
        # Get activation differences for this neuron across all PC ranges
        sae_neuron_diffs = sae_activation_matrix[:, neuron]
        dae_neuron_diffs = dae_activation_matrix[:, neuron]
        
        # Calculate rankings using argsort and flipping (1 = highest activation difference)
        sae_rankings[:, neuron] = np.argsort(np.argsort(-sae_neuron_diffs)) + 1
        dae_rankings[:, neuron] = np.argsort(np.argsort(-dae_neuron_diffs)) + 1
    
    # Create updated y-tick labels: PC Range updated from 0-based to 1-based indexing
    pc_labels = [f"{r[0]+1}-{r[1]}" for r in manipulated_neurons]
    
    # Create a grid with 1 row and 2 columns
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(),
                                        width_ratios=[1, 1], wspace=0.05)
    
    num_ranks = len(manipulated_neurons)
    blues = plt.cm.Blues_r(np.linspace(0, 1, num_ranks + 1))
    discrete_blues = ListedColormap(blues)
    reds = plt.cm.Reds_r(np.linspace(0, 1, num_ranks + 1))
    discrete_reds = ListedColormap(reds)
    
    bounds = [i + 0.5 for i in range(num_ranks + 1)]
    norm = BoundaryNorm(bounds, num_ranks)
    
    neuron_group_start_indices = [pair[0]+1 for pair in manipulated_neurons]
    
    display_indices = [idx for idx in neuron_group_start_indices if idx != 11]
    
    # SAE
    ax1 = plt.subplot(gs[0])
    
    sns.heatmap(sae_rankings, annot=False, cmap=discrete_blues, 
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=pc_labels, ax=ax1)
    
    ax1.set_yticklabels(pc_labels, rotation=0, fontsize=TICK_SIZE)
    
    # Add gray lines at neuron group boundaries
    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:  # Skip the first group
            ax1.axvline(x=start_idx-1, color='gray', linewidth=3.0)
    
    ax1.set_xticks([idx-1+0.5 for idx in display_indices])
    ax1.set_xticklabels(display_indices, fontsize=TICK_SIZE)
    
    # Colorbar
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_ticks(list(range(1, num_ranks + 1)))
    cbar1.set_ticklabels(list(range(1, num_ranks + 1)), fontsize=TICK_SIZE)
    cbar1.set_label('Ranking', fontsize=LABEL_SIZE, labelpad=10)
    cbar1.minorticks_off()
    cbar1.ax.invert_yaxis()
    
    ax1.set_title("AE", fontsize=LABEL_SIZE)
    ax1.set_xlabel("Neuron Index", fontsize=LABEL_SIZE)
    ax1.set_ylabel("PC Range", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    
    # DAE
    ax2 = plt.subplot(gs[1])
    
    sns.heatmap(dae_rankings, annot=False, cmap=discrete_reds, 
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=[], ax=ax2)
    
    # Add gray lines at neuron group boundaries
    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:
            ax2.axvline(x=start_idx-1, color='gray', linewidth=3.0)
    
    ax2.set_xticks([idx-1+0.5 for idx in display_indices])
    ax2.set_xticklabels(display_indices, fontsize=TICK_SIZE)
    
    # Colorbar
    cbar2 = ax2.collections[0].colorbar
    cbar2.set_ticks(list(range(1, num_ranks + 1)))
    cbar2.set_ticklabels(list(range(1, num_ranks + 1)), fontsize=TICK_SIZE)
    cbar2.set_label('Ranking', fontsize=LABEL_SIZE, labelpad=10)
    cbar2.minorticks_off()
    cbar2.ax.invert_yaxis()
    
    ax2.set_title("Dev-AE", fontsize=LABEL_SIZE)
    ax2.set_xlabel("Neuron Index", fontsize=LABEL_SIZE)


def plot_frequency_classification(ax, dataset="cifar"):
    """
    Plot frequency classification results (adapted from plot_frequency_classification_results)
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('mnist' or 'cifar')
    """
    results_file = f"Results/{dataset}_avg_frequency_classification.npy"
    std_file = f"Results/{dataset}_std_frequency_classification.npy"
    
    results = np.load(results_file, allow_pickle=True).item()
    std_devs = np.load(std_file, allow_pickle=True).item()
    
    p_values = None
    all_results_file = f"Results/{dataset}_all_frequency_classification.npy"
    all_results = np.load(all_results_file, allow_pickle=True).item()
    
    # Calculate p-values for each noise type
    _, p_clean = stats.ttest_rel(
        all_results['sae_clean_acc'],
        all_results['dae_clean_acc']
    )
    
    _, p_low = stats.ttest_rel(
        all_results['sae_low_freq_acc'],
        all_results['dae_low_freq_acc']
    )
    
    _, p_mid = stats.ttest_rel(
        all_results['sae_mid_freq_acc'],
        all_results['dae_mid_freq_acc']
    )
    
    _, p_high = stats.ttest_rel(
        all_results['sae_high_freq_acc'],
        all_results['dae_high_freq_acc']
    )
    
    p_values = {
        'clean': p_clean,
        'low_freq': p_low,
        'mid_freq': p_mid,
        'high_freq': p_high
    }
    
    # Calculate mean accuracies and standard deviations
    # for each noise type
    sae_accs = [
        results['sae_clean_acc'],
        results['sae_low_freq_acc'],
        results['sae_mid_freq_acc'],
        results['sae_high_freq_acc']
    ]
    dae_accs = [
        results['dae_clean_acc'],
        results['dae_low_freq_acc'],
        results['dae_mid_freq_acc'],
        results['dae_high_freq_acc']
    ]
    
    sae_errors = [
        std_devs.get('sae_clean_acc', 0),
        std_devs.get('sae_low_freq_acc', 0),
        std_devs.get('sae_mid_freq_acc', 0),
        std_devs.get('sae_high_freq_acc', 0)
    ]
    dae_errors = [
        std_devs.get('dae_clean_acc', 0),
        std_devs.get('dae_low_freq_acc', 0),
        std_devs.get('dae_mid_freq_acc', 0),
        std_devs.get('dae_high_freq_acc', 0)
    ]
    
    x = np.arange(4)
    width = 0.35
    
    sae_bars = ax.bar(x - width/2, sae_accs, width, label='AE', color='#1a7adb', 
                      yerr=sae_errors, capsize=5)
    dae_bars = ax.bar(x + width/2, dae_accs, width, label='Dev-AE', color='#e82817', 
                      yerr=dae_errors, capsize=5)
    
    ax.set_xlabel('Frequency Noise Type', fontsize=LABEL_SIZE)
    ax.set_ylabel('Classification Accuracy', fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Clean', 'Low\n(0-3)', 'Medium\n(4-7)', 'High\n(8-16)'], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), fontsize=LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if p_values is not None:
        p_list = [p_values.get('clean', 1), 
                 p_values.get('low_freq', 1), 
                 p_values.get('mid_freq', 1), 
                 p_values.get('high_freq', 1)]
        
        for i, p in enumerate(p_list):
            bar_height = max(sae_accs[i], dae_accs[i])
            y_pos = bar_height + max(sae_errors[i], dae_errors[i]) + 0.02
            
            if p < 0.05:
                ax.text(x[i], y_pos, '*', ha='center', va='bottom', fontsize=TICK_SIZE, weight='bold')
                line_y = y_pos - 0.005
                ax.hlines(y=line_y, xmin=x[i]-0.2, xmax=x[i]+0.2, linewidth=2, color='black')
            else:
                ax.text(x[i], y_pos, 'ns', ha='center', va='bottom', fontsize=TICK_SIZE, weight='bold')
                line_y = y_pos - 0.005
                ax.hlines(y=line_y, xmin=x[i]-0.2, xmax=x[i]+0.2, linewidth=2, color='black')


def plot_neuron_importance(ax, dataset="cifar"):
    """
    Plot neuron group importance (adapted from plot_grouped_importance)
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [6, 12, 18, 28, 48, 90, 128]
    else:
        neuron_groups = [6, 12, 18, 28, 48, 90, 128]
    
    result_file = f"Results/{dataset}_neuron_importance.npy"
    avg_results = np.load(result_file, allow_pickle=True).item()
    
    sae_importance = avg_results['sae_importance']
    dae_importance = avg_results['dae_importance']
    neuron_groups = avg_results['neuron_groups']
    
    sae_group_importance = get_neuron_group_importance(sae_importance, neuron_groups)
    dae_group_importance = get_neuron_group_importance(dae_importance, neuron_groups)
    
    sae_group_error = np.zeros_like(sae_group_importance)
    dae_group_error = np.zeros_like(dae_group_importance)
    
    if 'all_sae_group_importance' in avg_results:
        all_sae_group = np.array(avg_results['all_sae_group_importance'])
        all_dae_group = np.array(avg_results['all_dae_group_importance'])
        num_models = len(all_sae_group)
        
        if num_models > 1:
            sae_group_error = np.std(all_sae_group, axis=0) / np.sqrt(num_models)
            dae_group_error = np.std(all_dae_group, axis=0) / np.sqrt(num_models)
    
    start_indices = [1] + [neuron_groups[i-1] + 1 for i in range(1, len(neuron_groups))]
    x_labels = [f"{start}-{end}" for start, end in zip(start_indices, neuron_groups)]
    
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(),
                                         width_ratios=[1, 1], wspace=0.05)
    
    ax.axis('off')
    
    x_indices = np.arange(len(neuron_groups))
    
    # SAE
    ax1 = plt.subplot(gs[0])
    sae_bars = ax1.bar(x_indices, sae_group_importance, color='#1a7adb', 
                       yerr=sae_group_error, capsize=5, ecolor='black')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=90)
    ax1.set_ylabel('Classification Importance', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # DAE
    ax2 = plt.subplot(gs[1])
    dae_bars = ax2.bar(x_indices, dae_group_importance, color='#e82817',
                       yerr=dae_group_error, capsize=5, ecolor='black')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=90)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')

    max_val = max(
        max(np.array(sae_group_importance) + np.array(sae_group_error)), 
        max(np.array(dae_group_importance) + np.array(dae_group_error))
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)

    ax.text(0.5, -0.4, 'Neuron Groups', ha='center', fontsize=LABEL_SIZE, transform=ax.transAxes)



if DATASET.lower() == "mnist":
    manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    neuron_groups = [6, 12, 18, 28, 48, 90, 128]
else:
    manipulated_neurons = [(0, 6), (6, 10), (10, 16), (16, 28), (28, 48), (48, 90), (90, 128)]
    neuron_groups = [6, 12, 18, 28, 48, 90, 128]
    
compute_pc_noise_analysis(NUM_MODELS, manipulated_neurons, DATASET, BASE_PATH)
compute_average_frequency_classification(NUM_MODELS, DATASET, BASE_PATH, noise_scale=1.0)
compute_neuron_importance(NUM_MODELS, DATASET, BASE_PATH, neuron_groups)

create_figure_3(DATASET)