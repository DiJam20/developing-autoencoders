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
    fig = plt.figure(figsize=(20, 12))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.8, 1], wspace=0.3, hspace=0.4)
    
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
    ax_bottom_left.text(-0.2, 1.05, 'B', transform=ax_bottom_left.transAxes, 
                        fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom_right.text(-0.2, 1.05, 'C', transform=ax_bottom_right.transAxes, 
                         fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"paper_results/figures/png/{dataset}_figure_3.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"paper_results/figures/pdf/{dataset}_figure_3.pdf", bbox_inches='tight')
    plt.close()


def plot_pc_rankings(ax, dataset="cifar"):
    """
    Plot PC noise impact rankings heatmap (adapted from create_ranking_heatmaps)
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('mnist' or 'cifar')
    """
    ax.axis('off')
        
    results = np.load(f"paper_results/{dataset}_pc_noise.npy", allow_pickle=True).item()
    
    if dataset.lower() == "mnist":
        manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    else:
        # 6, 10, 17, 29, 50, 85, 128
        manipulated_neurons = [(0, 6), (6, 10), (10, 17), (17, 29), (29, 50), (50, 85), (85, 128)]
    
    sample_data = next(iter(results.values()))[0]
    tuple_len = len(sample_data) if isinstance(sample_data, (list, tuple)) else 2
    num_neurons = len(sample_data[0]) if isinstance(sample_data, (list, tuple)) else len(sample_data)

    include_pca = tuple_len >= 3

    sae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))
    pca_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons)) if include_pca else None
    dae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))

    # Calculate mean activation differences for each PC range and each neuron
    for i, pc_range in enumerate(manipulated_neurons):
        # Get all runs for this PC range
        runs = results[pc_range]

        # Stack all differences for this PC range
        sae_diffs = np.vstack([run[0] for run in runs])
        if include_pca:
            pca_diffs = np.vstack([run[1] for run in runs])
            dae_diffs = np.vstack([run[2] for run in runs])
        else:
            dae_diffs = np.vstack([run[1] for run in runs])
            pca_diffs = None

        # Calculate mean across runs
        sae_mean = np.mean(sae_diffs, axis=0)
        dae_mean = np.mean(dae_diffs, axis=0)
        pca_mean = np.mean(pca_diffs, axis=0) if include_pca else None

        # Store in matrices
        sae_activation_matrix[i, :] = sae_mean
        dae_activation_matrix[i, :] = dae_mean
        if include_pca:
            pca_activation_matrix[i, :] = pca_mean

    # Calculate rankings for each neuron (1 = highest activation difference)
    sae_rankings = np.zeros_like(sae_activation_matrix, dtype=int)
    dae_rankings = np.zeros_like(dae_activation_matrix, dtype=int)
    pca_rankings = np.zeros_like(pca_activation_matrix, dtype=int) if include_pca else None

    for neuron in range(num_neurons):
        # Get activation differences for this neuron across all PC ranges
        sae_neuron_diffs = sae_activation_matrix[:, neuron]
        dae_neuron_diffs = dae_activation_matrix[:, neuron]
        pca_neuron_diffs = pca_activation_matrix[:, neuron] if include_pca else None

        # Calculate rankings using argsort and flipping (1 = highest activation difference)
        sae_rankings[:, neuron] = np.argsort(np.argsort(-sae_neuron_diffs)) + 1
        dae_rankings[:, neuron] = np.argsort(np.argsort(-dae_neuron_diffs)) + 1
        if include_pca:
            pca_rankings[:, neuron] = np.argsort(np.argsort(-pca_neuron_diffs)) + 1
    
    # Create updated y-tick labels: PC Range updated from 0-based to 1-based indexing
    pc_labels = [f"{r[0]+1}-{r[1]}" for r in manipulated_neurons]
    
    # Create a grid with 1 row and 2 columns
    n_cols = 3 if include_pca else 2
    gs = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=ax.get_subplotspec(),
                                        width_ratios=[1] * n_cols, wspace=0.15)
    
    num_ranks = len(manipulated_neurons)
    blues = plt.cm.Blues_r(np.linspace(0, 1, num_ranks + 1))
    discrete_blues = ListedColormap(blues)
    greens = plt.cm.Greens_r(np.linspace(0, 1, num_ranks + 1))
    discrete_greens = ListedColormap(greens)
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
    ax1.set_xticklabels(display_indices, fontsize=TICK_SIZE, rotation=90)
    
    # Colorbar
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_ticks(list(range(1, num_ranks + 1)))
    cbar1.set_ticklabels(list(range(1, num_ranks + 1)), fontsize=TICK_SIZE)
    cbar1.set_label('Ranking', fontsize=LABEL_SIZE, labelpad=10)
    cbar1.minorticks_off()
    cbar1.ax.invert_yaxis()
    
    ax1.set_title("AE", fontsize=LABEL_SIZE)
    ax1.set_xlabel("Neuron Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Perturbed PC Range", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE, width=2)
    ax1.tick_params(axis='x', width=2)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    ax2 = plt.subplot(gs[1])
    sns.heatmap(pca_rankings, annot=False, cmap=discrete_greens,
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=[], ax=ax2)

    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:
            ax2.axvline(x=start_idx-1, color='gray', linewidth=3.0)

    ax2.set_xticks([idx-1+0.5 for idx in display_indices])
    ax2.set_xticklabels(display_indices, fontsize=TICK_SIZE, rotation=90)

    cbar2 = ax2.collections[0].colorbar
    cbar2.set_ticks(list(range(1, num_ranks + 1)))
    cbar2.set_ticklabels(list(range(1, num_ranks + 1)), fontsize=TICK_SIZE)
    cbar2.set_label('Ranking', fontsize=LABEL_SIZE, labelpad=10)
    cbar2.minorticks_off()
    cbar2.ax.invert_yaxis()

    ax2.set_title("PCA-AE", fontsize=LABEL_SIZE)
    ax2.set_xlabel("Neuron Index", fontsize=LABEL_SIZE)
    ax2.tick_params(axis='x', width=2)
    ax2.tick_params(axis='y', width=2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    ax3 = plt.subplot(gs[2])
    sns.heatmap(dae_rankings, annot=False, cmap=discrete_reds, 
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=[], ax=ax3)
    
    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:
            ax3.axvline(x=start_idx-1, color='gray', linewidth=3.0)
    
    ax3.set_xticks([idx-1+0.5 for idx in display_indices])
    ax3.set_xticklabels(display_indices, fontsize=TICK_SIZE, rotation=90)
    
    cbar3 = ax3.collections[0].colorbar
    cbar3.set_ticks(list(range(1, num_ranks + 1)))
    cbar3.set_ticklabels(list(range(1, num_ranks + 1)), fontsize=TICK_SIZE)
    cbar3.set_label('Ranking', fontsize=LABEL_SIZE, labelpad=10)
    cbar3.minorticks_off()
    cbar3.ax.invert_yaxis()

    ax3.set_title("Dev-AE", fontsize=LABEL_SIZE)
    ax3.set_xlabel("Neuron Index", fontsize=LABEL_SIZE)
    ax3.tick_params(axis='x', width=2)
    ax3.tick_params(axis='y', width=2)
    for spine in ax3.spines.values():
        spine.set_linewidth(2)


def load_and_calculate_pvalues(dataset):
    """
    Load results and calculate p-values for each noise type.
    
    Args:
        dataset: Dataset name ('mnist' or 'cifar')
        
    Returns:
        Dictionary with p-values for each noise type
    """
    # Load all individual results (not just averages)
    all_results_file = f"paper_results/{dataset}_all_frequency_classification.npy"
    
    try:
        all_results = np.load(all_results_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find {all_results_file}")
        print("Make sure you've run the main script to generate all results.")
        return None
    
    # Calculate p-values for each noise type using arrays of individual measurements
    p_values = {}

    # --- SAE vs DAE ---
    _, p_values['clean_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_clean_acc'],
        all_results['dae_clean_acc']
    )
    _, p_values['low_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_low_freq_acc'],
        all_results['dae_low_freq_acc']
    )
    _, p_values['mid_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_mid_freq_acc'],
        all_results['dae_mid_freq_acc']
    )
    _, p_values['high_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_high_freq_acc'],
        all_results['dae_high_freq_acc']
    )

    # --- SAE vs PCA-AE ---
    _, p_values['clean_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_clean_acc'],
        all_results['pca_ae_clean_acc']
    )
    _, p_values['low_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_low_freq_acc'],
        all_results['pca_ae_low_freq_acc']
    )
    _, p_values['mid_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_mid_freq_acc'],
        all_results['pca_ae_mid_freq_acc']
    )
    _, p_values['high_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_high_freq_acc'],
        all_results['pca_ae_high_freq_acc']
    )

    # --- DAE vs PCA-AE ---
    _, p_values['clean_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_clean_acc'],
        all_results['pca_ae_clean_acc']
    )
    _, p_values['low_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_low_freq_acc'],
        all_results['pca_ae_low_freq_acc']
    )
    _, p_values['mid_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_mid_freq_acc'],
        all_results['pca_ae_mid_freq_acc']
    )
    _, p_values['high_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_high_freq_acc'],
        all_results['pca_ae_high_freq_acc']
    )
        
    return p_values


def add_significance_marker(x1, x2, y_start, p_value, ax):
    """
    Adds a significance marker (bracket with star) between two bars.
    """
    if p_value >= 0.05:
        return  # Don't draw for non-significant results

    # Determine star text
    if p_value < 0.001:
        star_text = '***'
    elif p_value < 0.01:
        star_text = '**'
    else:
        star_text = '*'

    # Bracket line
    ax.plot([x1, x1, x2, x2], [y_start, y_start + 0.01, y_start + 0.01, y_start], lw=1, c='black')
    
    # Star text
    ax.text((x1 + x2) * 0.5, y_start + 0.01, star_text, ha='center', va='bottom', color='black', fontsize=TICK_SIZE)


def plot_frequency_classification(ax, dataset="cifar"):
    """
    Plot frequency classification results (adapted from plot_frequency_classification_results)
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('mnist' or 'cifar')
    """
    results_file = f"paper_results/{dataset}_frequency_classification_summary.npy"
    
    results = np.load(results_file, allow_pickle=True).item()
    
    p_values = load_and_calculate_pvalues(dataset) #TODO

    # Calculate mean accuracies and standard deviations
    # for each noise type
    sae_accs = [
        results['sae']['clean']['avg'],
        results['sae']['low_freq']['avg'],
        results['sae']['mid_freq']['avg'],
        results['sae']['high_freq']['avg']
    ]
    pca_ae_accs = [
        results['pca_ae']['clean']['avg'],
        results['pca_ae']['low_freq']['avg'],
        results['pca_ae']['mid_freq']['avg'],
        results['pca_ae']['high_freq']['avg']
    ]
    dae_accs = [
        results['dae']['clean']['avg'],
        results['dae']['low_freq']['avg'],
        results['dae']['mid_freq']['avg'],
        results['dae']['high_freq']['avg']
    ]
    
    sae_errors = [
        results['sae']['clean']['std'],
        results['sae']['low_freq']['std'],
        results['sae']['mid_freq']['std'],
        results['sae']['high_freq']['std']
    ]
    pca_ae_errors = [
        results['pca_ae']['clean']['std'],
        results['pca_ae']['low_freq']['std'],
        results['pca_ae']['mid_freq']['std'],
        results['pca_ae']['high_freq']['std']
    ]
    dae_errors = [
        results['dae']['clean']['std'],
        results['dae']['low_freq']['std'],
        results['dae']['mid_freq']['std'],
        results['dae']['high_freq']['std']
    ]

    # Get individual datapoints for each model
    sae_all = [
        results['sae']['clean']['all'],
        results['sae']['low_freq']['all'],
        results['sae']['mid_freq']['all'],
        results['sae']['high_freq']['all']
    ]
    pca_ae_all = [
        results['pca_ae']['clean']['all'],
        results['pca_ae']['low_freq']['all'],
        results['pca_ae']['mid_freq']['all'],
        results['pca_ae']['high_freq']['all']
    ]
    dae_all = [
        results['dae']['clean']['all'],
        results['dae']['low_freq']['all'],
        results['dae']['mid_freq']['all'],
        results['dae']['high_freq']['all']
    ]


    sae_errors = [results['sae']['clean']['std'], results['sae']['low_freq']['std'], results['sae']['mid_freq']['std'], results['sae']['high_freq']['std']]
    pca_ae_errors = [results['pca_ae']['clean']['std'], results['pca_ae']['low_freq']['std'], results['pca_ae']['mid_freq']['std'], results['pca_ae']['high_freq']['std']]
    dae_errors = [results['dae']['clean']['std'], results['dae']['low_freq']['std'], results['dae']['mid_freq']['std'], results['dae']['high_freq']['std']]

    x = np.arange(4)
    width = 0.25
    
    # Create the bar plots with zorder=0
    sae_bars = ax.bar(x - width, sae_accs, width, label='AE', color='#1a7adb', 
                      zorder=0)
    pca_ae_bars = ax.bar(x, pca_ae_accs, width, label='PCA-AE', color='#00a65a',
                         zorder=0)
    dae_bars = ax.bar(x + width, dae_accs, width, label='Dev-AE', color='#e82817', 
                      zorder=0)
    
    # Add error bars separately with zorder=2
    ax.errorbar(x - width, sae_accs, yerr=sae_errors, fmt='none', ecolor='black', 
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=2)
    ax.errorbar(x, pca_ae_accs, yerr=pca_ae_errors, fmt='none', ecolor='black',
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=2)
    ax.errorbar(x + width, dae_accs, yerr=dae_errors, fmt='none', ecolor='black',
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=2)
    
    # Add individual datapoints as scatter plots with lower zorder
    for i in range(4):
        # Add jitter to x-coordinates for better visibility
        jitter = 0.035
        
        # SAE datapoints
        x_sae = np.random.normal(x[i] - width, jitter, size=len(sae_all[i]))
        ax.scatter(x_sae, sae_all[i], color='#1a7adb', s=40, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)
        
        # PCA-AE datapoints
        x_pca = np.random.normal(x[i], jitter, size=len(pca_ae_all[i]))
        ax.scatter(x_pca, pca_ae_all[i], color='#00a65a', s=40, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)

        # DAE datapoints
        x_dae = np.random.normal(x[i] + width, jitter, size=len(dae_all[i]))
        ax.scatter(x_dae, dae_all[i], color='#e82817', s=40, alpha=0.6, zorder=1, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Frequency Noise Type', fontsize=LABEL_SIZE)
    ax.set_ylabel('Classification Accuracy', fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Clean', 'Low', 'Medium', 'High'], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE, width=2)
    ax.tick_params(axis='x', width=2)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), fontsize=LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    if p_values is not None:
        p_keys_sae_dae = ['clean_sae_vs_dae', 'low_freq_sae_vs_dae', 'mid_freq_sae_vs_dae', 'high_freq_sae_vs_dae']
        p_keys_sae_pca = ['clean_sae_vs_pca', 'low_freq_sae_vs_pca', 'mid_freq_sae_vs_pca', 'high_freq_sae_vs_pca']
        p_keys_dae_pca = ['clean_dae_vs_pca', 'low_freq_dae_vs_pca', 'mid_freq_dae_vs_pca', 'high_freq_dae_vs_pca']

        for i in range(4):
            # Determine the top of the bars for positioning the markers
            all_heights = [sae_accs[i] + sae_errors[i], pca_ae_accs[i] + pca_ae_errors[i], dae_accs[i] + dae_errors[i]]
            y_base = max(all_heights) + 0.02

            # SAE vs DAE
            add_significance_marker(x[i] - width, x[i] + width, y_base + 0.05, p_values.get(p_keys_sae_dae[i], 1), ax)
            # SAE vs PCA-AE
            add_significance_marker(x[i] - width, x[i], y_base, p_values.get(p_keys_sae_pca[i], 1), ax)
            # DAE vs PCA-AE
            add_significance_marker(x[i], x[i] + width, y_base, p_values.get(p_keys_dae_pca[i], 1), ax)

    # Adjust y-axis limit to make space for markers
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)


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
        neuron_groups = [6, 10, 17, 29, 50, 85, 128]
    
    result_file = f"paper_results/{dataset}_neuron_importance.npy"
    avg_results = np.load(result_file, allow_pickle=True).item()

    all_sae_group = np.array(avg_results['all_sae_group_importance'])
    all_pca_group = np.array(avg_results['all_pca_ae_group_importance'])
    all_dae_group = np.array(avg_results['all_dae_group_importance'])

    # Normalize each model's run so that it sums to 1
    sae_norm_all = all_sae_group / all_sae_group.sum(axis=1, keepdims=True)
    pca_norm_all = all_pca_group / all_pca_group.sum(axis=1, keepdims=True)
    dae_norm_all = all_dae_group / all_dae_group.sum(axis=1, keepdims=True)

    # Now calculate mean and std dev from the normalized data
    sae_importance_normal = np.mean(sae_norm_all, axis=0)
    pca_importance_normal = np.mean(pca_norm_all, axis=0)
    dae_importance_normal = np.mean(dae_norm_all, axis=0)

    sae_group_error = np.std(sae_norm_all, axis=0)
    pca_group_error = np.std(pca_norm_all, axis=0)
    dae_group_error = np.std(dae_norm_all, axis=0)

    # Labels per neuron group
    start_indices = [1] + [neuron_groups[i-1] + 1 for i in range(1, len(neuron_groups))]
    x_labels = [f"{start}-{end}" for start, end in zip(start_indices, neuron_groups)]

    x_indices = np.arange(len(neuron_groups))
    width = 0.25

    ax.bar(
        x_indices-width,
        sae_importance_normal,
        width,
        color='#1a7adb',
        yerr=sae_group_error,
        capsize=5,
        ecolor='black',
    )
    ax.bar(
        x_indices,
        pca_importance_normal,
        width,
        color='#00a65a',
        yerr=pca_group_error,
        capsize=5,
        ecolor='black',
    )
    ax.bar(
        x_indices+width,
        dae_importance_normal,
        width,
        color='#e82817',
        yerr=dae_group_error,
        capsize=5,
        ecolor='black',
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=TICK_SIZE, rotation=45)
    ax.set_xlabel('Neuron Groups', fontsize=LABEL_SIZE)
    ax.set_ylabel('Classification Importance', fontsize=LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE, width=2)
    ax.tick_params(axis='x', width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_ylim(0,)


if DATASET.lower() == "mnist":
    manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    neuron_groups = [6, 12, 18, 28, 48, 90, 128]
else:
    manipulated_neurons = [(0, 6), (6, 10), (10, 17), (17, 29), (29, 50), (50, 85), (85, 128)]
    neuron_groups = [6, 10, 17, 29, 50, 85, 128]
    
# compute_pc_noise_analysis(NUM_MODELS, manipulated_neurons, DATASET, BASE_PATH)
# compute_average_frequency_classification(NUM_MODELS, DATASET, BASE_PATH, noise_scale=1.0, create_plots=False)
# compute_neuron_importance(NUM_MODELS, DATASET, BASE_PATH, neuron_groups)

create_figure_3(DATASET)