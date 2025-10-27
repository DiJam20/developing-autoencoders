import os
from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from rf_computation import compute_final_rfs
from power_spectra_paper import compute_power_spectra


LABEL_SIZE = 22
TICK_SIZE = 22
LEGEND_SIZE = 16
TITLE_SIZE = 26

MODEL_IDX_FOR_EXAMPLE_RF = 3

mnist_size_ls = [4, 4, 4, 4, 4, 10,
                10, 10, 10, 10, 16, 16,
                16, 16, 16, 16, 16, 24,
                24, 24, 24, 24, 24, 24, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32,]

cifar_size_ls = [6] * 6 + [10] * 6 + [17] * 7 + [29] * 7 + [50] * 8 + [85] * 8 + [128] * 18

# Neuron group endpoints
neuron_groups = [6, 10, 17, 29, 50, 85, 128]

# def compute_rgb_jsd_multi(model_data, rf_range, bins=50, hist_range=(0, 1)):
#     """
#     Compute mean JSD between RGB channels for a set of receptive fields.
    
#     Args:
#         model_data: Array of shape (num_neurons, channels, height, width)
#         rf_range: Range of RF indices to analyze
#         bins: Number of histogram bins
#         hist_range: Range for histogram
    
#     Returns:
#         Mean JSD across all pairwise channel comparisons
#     """
#     subset = model_data[rf_range, :, :, :]
    
#     # Compute histograms for each channel
#     hists = []
#     for c in range(3):
#         values = subset[:, c, :, :].ravel()
#         hist, _ = np.histogram(values, bins=bins, range=hist_range, density=True)
#         hist = hist + 1e-10  # Avoid zero probabilities
#         hist = hist / hist.sum()  # Normalize
#         hists.append(hist)
    
#     # Compute pairwise JSD
#     jsd_values = []
#     for i in range(3):
#         for j in range(i+1, 3):
#             jsd = jensenshannon(hists[i], hists[j])
#             jsd_values.append(jsd)
    
#     return np.mean(jsd_values)

def js_divergence_multi(dists, base=np.e):
    """
    Compute Jensen-Shannon divergence among N distributions.
    dists: list or array of shape (N, bins)
    Returns scalar JSD.
    """
    dists = np.asarray(dists, dtype=np.float64)
    eps = 1e-12
    dists = dists + eps
    dists = dists / dists.sum(axis=1, keepdims=True)  # normalize each
    M = np.mean(dists, axis=0)
    H_M = entropy(M, base=base)
    H_P = np.mean([entropy(p, base=base) for p in dists])
    return H_M - H_P

# --- Compute JSD for selected receptive field range ---
def compute_rgb_jsd_multi(model_data, rf_range, bins=50, hist_range=(0, 1)):
    """
    model_data: (128, 3, 32, 32)
    rf_range: Python range or list of receptive field indices to include
    """
    subset = model_data[rf_range, :, :, :]  # restrict receptive fields
    channels = [subset[:, c, :, :].ravel() for c in range(3)]
    hists = [np.histogram(ch, bins=bins, range=hist_range, density=True)[0] for ch in channels]
    return js_divergence_multi(hists)



def create_figure_2(dataset='cifar', num_models=2, final_epoch=59):
    """
    Create Figure 2 with four subplots:
    1. Example RFs (top left)
    2. RGB histograms (top right)
    3. Power spectra (bottom left)
    4. JSD summary (bottom right)
    
    Args:
        dataset: Dataset name ('cifar')
        num_models: Number of models to use
        final_epoch: Final epoch to use for analysis
    """
    fig = plt.figure(figsize=(26, 15))
    
    # 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1.1], height_ratios=[5.5, 2], 
                          wspace=0.25, hspace=0.3)
    
    gs_top_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 0])
    gs_top_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 1])
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 0])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1])
    
    ax_rfs = plt.subplot(gs_top_left[0])
    ax_histograms = plt.subplot(gs_top_right[0])
    ax_spectra = plt.subplot(gs_bottom_left[0])
    ax_jsd_summary = plt.subplot(gs_bottom_right[0])
    
    plot_example_rfs(ax_rfs, dataset, final_epoch)
    plot_rgb_histograms(ax_histograms, dataset, num_models)
    plot_power_spectra(ax_spectra, dataset, num_models, final_epoch)
    plot_jsd_summary(ax_jsd_summary, dataset, num_models)

    # Add figure labels
    ax_rfs.text(-0.08, 1, 'A', transform=ax_rfs.transAxes, 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax_histograms.text(-0.25, 1, 'C', transform=ax_histograms.transAxes, 
                       fontsize=TITLE_SIZE, fontweight='bold')
    ax_spectra.text(-0.08, 1.18, 'B', transform=ax_spectra.transAxes, 
                    fontsize=TITLE_SIZE, fontweight='bold')
    ax_jsd_summary.text(-0.25, 1.18, 'D', transform=ax_jsd_summary.transAxes, 
                       fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"paper_results/figures/png/{dataset}_figure_2.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"paper_results/figures/pdf/{dataset}_figure_2.pdf", bbox_inches='tight')
    plt.close()


def plot_example_rfs(ax, dataset='cifar', epoch_idx=59):
    """
    Plot example receptive fields for the first subplot.
    Show four RFs per group for both AE and Dev-AE in a 2x2 grid.
    Uses original 0-1 clipping and improved spacing with centered titles.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        epoch_idx: Epoch index to use
    """
    # Load RFs
    sae_rfs = np.load(f'paper_results/{dataset}_sae_final_rfs.npy')
    pca_init_sae_rfs = np.load(f'paper_results/{dataset}_pca-ae_final_rfs.npy')
    dae_rfs = np.load(f'paper_results/{dataset}_dev-ae_final_rfs.npy')

    # Get the neuron indices for each group
    group_boundaries = [0] + neuron_groups
    
    ax.axis('off')
    
    num_groups = len(group_boundaries) - 1
    
    # Grid with 4 rows: AE, PCA-init-SAE, Dev-AE, arrow
    outer_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=ax.get_subplotspec(),
                                                 height_ratios=[1, 1, 1, 0.15], 
                                                 hspace=0.1)
    
    grid_wspace = 0.15
    rf_wspace = 0.1
    rf_hspace = 0
    
    # Grids for AE, PCA-init-SAE, and Dev-AE different neuron groups
    ae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                     subplot_spec=outer_grid[0],
                                                     wspace=grid_wspace)
    pca_init_sae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                                subplot_spec=outer_grid[1],
                                                                wspace=grid_wspace)
    devae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                        subplot_spec=outer_grid[2],
                                                        wspace=grid_wspace)
    
    arrow_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[3])
    
    group_labels = []
    for i in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[i]
        end_idx = group_boundaries[i+1]
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
    
    row_labels = ["AE", "PCA-AE", "Dev-AE"]

    # Separate axis for the AE, PCA-AE, and Dev-AE labels
    ae_label_ax = plt.subplot(outer_grid[0])
    ae_label_ax.axis('off')
    ae_label_ax.text(-0.05, 0.5, row_labels[0], rotation=90, 
                     transform=ae_label_ax.transAxes, 
                     ha='center', va='center', fontsize=LABEL_SIZE)
    pca_init_sae_label_ax = plt.subplot(outer_grid[1])
    pca_init_sae_label_ax.axis('off')
    pca_init_sae_label_ax.text(-0.05, 0.5, row_labels[1], rotation=90, 
                        transform=pca_init_sae_label_ax.transAxes, 
                        ha='center', va='center', fontsize=LABEL_SIZE)
    devae_label_ax = plt.subplot(outer_grid[2])
    devae_label_ax.axis('off')
    devae_label_ax.text(-0.05, 0.5, row_labels[2], rotation=90, 
                        transform=devae_label_ax.transAxes, 
                        ha='center', va='center', fontsize=LABEL_SIZE)
    
    # AE RFs
    for col_group in range(num_groups):
        # Grid with 3 rows: title, 2x2 RFs (2 columns)
        ae_rf_grid = gridspec.GridSpecFromSubplotSpec(3, 2, 
                                                    subplot_spec=ae_groups_grid[col_group],
                                                    height_ratios=[0.2, 1, 1],
                                                    wspace=rf_wspace, hspace=rf_hspace)
        
        start_idx = group_boundaries[col_group]
        
        # Neuron group title
        group_title_ax = plt.subplot(ae_rf_grid[0, :])
        group_title_ax.set_title(group_labels[col_group], fontsize=LABEL_SIZE)
        
        group_title_ax.axis('off')
        
        rf_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for rf_idx, (row, col) in enumerate(rf_positions):
            neuron_idx = start_idx + rf_idx
            
            curr_ax = plt.subplot(ae_rf_grid[row, col])
            
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            # Get the RF to display - AE
            if neuron_idx < len(sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF]):
                rf_data = sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, neuron_idx]
                
                if dataset.lower() == 'cifar':
                    rf_data = np.transpose(rf_data, (1, 2, 0))
                    rf_data = np.clip(rf_data, 0, 1)
                
                curr_ax.imshow(rf_data)
    
    # PCA-init-SAE RFs
    for col_group in range(num_groups):
        # Grid with 3 rows: title, 2x2 RFs (2 columns)
        pca_init_sae_rf_grid = gridspec.GridSpecFromSubplotSpec(3, 2, 
                                                    subplot_spec=pca_init_sae_groups_grid[col_group],
                                                    height_ratios=[0.2, 1, 1],
                                                    wspace=rf_wspace, hspace=rf_hspace)
        
        start_idx = group_boundaries[col_group]
        
        rf_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for rf_idx, (row, col) in enumerate(rf_positions):
            neuron_idx = start_idx + rf_idx
            
            curr_ax = plt.subplot(pca_init_sae_rf_grid[row, col])
            
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            # Get the RF to display - PCA-init-SAE
            if neuron_idx < len(pca_init_sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF]):
                rf_data = pca_init_sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, neuron_idx]
                
                if dataset.lower() == 'cifar':
                    rf_data = np.transpose(rf_data, (1, 2, 0))
                    rf_data = np.clip(rf_data, 0, 1)
                
                curr_ax.imshow(rf_data)
    
    # Dev-AE RFs
    for col_group in range(num_groups):
        # Grid with 3 rows: title, 2x2 RFs (2 columns)
        devae_rf_grid = gridspec.GridSpecFromSubplotSpec(3, 2, 
                                                       subplot_spec=devae_groups_grid[col_group],
                                                       height_ratios=[0.2, 1, 1],
                                                       wspace=rf_wspace, hspace=rf_hspace)
        
        start_idx = group_boundaries[col_group]
        
        rf_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for rf_idx, (row, col) in enumerate(rf_positions):
            neuron_idx = start_idx + rf_idx
            
            curr_ax = plt.subplot(devae_rf_grid[row, col])
            
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            if neuron_idx < len(dae_rfs[MODEL_IDX_FOR_EXAMPLE_RF]):
                rf_data = dae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, neuron_idx]
                
                if dataset.lower() == 'cifar':
                    rf_data = np.transpose(rf_data, (1, 2, 0))
                    rf_data = np.clip(rf_data, 0, 1)
                
                curr_ax.imshow(rf_data)
    
    # Arrow
    arrow_ax = plt.subplot(arrow_grid[0])
    arrow_ax.axis('off')
    
    n_groups = len(group_boundaries) - 1
    colors = plt.cm.cool(np.linspace(0, 1, n_groups))
    
    arrow_start = 0.3
    arrow_end = 0.7
    
    # Use mutiple segments for the arrow
    segments = len(neuron_groups) - 1
    segment_length = (arrow_end - arrow_start) / segments
    
    for i in range(segments):
        pos_start = arrow_start + i * segment_length
        pos_end = pos_start + segment_length
        
        color_idx = int(i / segments * (n_groups - 1))
        color = colors[color_idx]
        
        # Segments of arrow (if last, then arrow)
        if i == segments - 1:
            arrow_ax.annotate('', 
                             xy=(pos_end, 0.5), 
                             xytext=(pos_start, 0.5),
                             arrowprops=dict(arrowstyle="->", lw=4, color=color),
                             xycoords='axes fraction', 
                             textcoords='axes fraction')
        else:
            arrow_ax.annotate('', 
                             xy=(pos_end, 0.5), 
                             xytext=(pos_start, 0.5),
                             arrowprops=dict(arrowstyle="-", lw=4, color=color),
                             xycoords='axes fraction', 
                             textcoords='axes fraction')
    
    # "Early" and "Late" text
    arrow_ax.text(arrow_start-0.035, 0.5, 'Early', fontsize=LABEL_SIZE, ha='center', va='center',
                 transform=arrow_ax.transAxes)
    arrow_ax.text(arrow_end+0.025, 0.5, 'Late', fontsize=LABEL_SIZE, ha='center', va='center',
                 transform=arrow_ax.transAxes)


def plot_rgb_histograms(ax, dataset='cifar', num_models=2):
    """
    Plot RGB histograms for AE, PCA-AE, and Dev-AE models.
    Shows 2 example models per architecture type with JSD values.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        num_models: Number of models to show examples from
    """
    # Load RFs
    sae_rfs = np.load(f'paper_results/{dataset}_sae_final_rfs.npy')
    pca_init_sae_rfs = np.load(f'paper_results/{dataset}_pca-ae_final_rfs.npy')
    dae_rfs = np.load(f'paper_results/{dataset}_dev-ae_final_rfs.npy')
    
    ax.axis('off')

    # 4 rows (AE, PCA-AE, Dev-AE) × 2 columns (2 model examples)
    grid = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=ax.get_subplotspec(),
                                            height_ratios=[1,1,1,0.01],
                                            hspace=0.45, wspace=0.3)
    
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    bins = 50
    hist_range = (-20, 20)
    
    # RF range to analyze (all neurons)
    rf_range = slice(0, None)
    
    model_types = [
        ('AE', sae_rfs),
        ('PCA-AE', pca_init_sae_rfs),
        ('Dev-AE', dae_rfs)
    ]
    
    # Plot histograms for each model type
    for row_idx, (model_name, model_rfs) in enumerate(model_types):
        for col_idx in range(min(2, num_models)):
            subplot_ax = plt.subplot(grid[row_idx, col_idx])
            
            if model_name == 'AE' and col_idx == 0:
                model_data = model_rfs[1]
            elif model_name == 'AE' and col_idx == 1:
                model_data = model_rfs[0]
            elif model_name == 'PCA-AE' and col_idx == 1:
                model_data = model_rfs[3]
            else:
                model_data = model_rfs[col_idx]
            
            # Transpose from (neurons, channels, h, w) to (neurons, h, w, channels) if needed
            if dataset.lower() == 'cifar' and model_data.shape[1] == 3:
                model_data = np.transpose(model_data, (0, 2, 3, 1))
            
            # Plot histograms for each RGB channel
            subset = model_data[rf_range]
            for c, color in enumerate(colors):
                values = subset[:, :, :, c].ravel()
                subplot_ax.hist(values, bins=bins, range=hist_range, 
                               color=color, alpha=0.5, label=channel_names[c])
            
            # Compute JSD
            # Convert back for JSD computation
            if dataset.lower() == 'cifar':
                model_data_for_jsd = np.transpose(model_data, (0, 3, 1, 2))
            else:
                model_data_for_jsd = model_data
                
            jsd_mean = compute_rgb_jsd_multi(model_data_for_jsd, rf_range, 
                                            bins=bins, hist_range=hist_range)
            
            subplot_ax.set_title(f"JSD≈{jsd_mean:.3f}", fontsize=LEGEND_SIZE, y=0.94)
            subplot_ax.tick_params(labelsize=TICK_SIZE-4, width=2)
            if model_name == 'AE':
                subplot_ax.set_ylim(0, 8000)
            elif model_name == 'PCA-AE':
                subplot_ax.set_ylim(0, 6000)
            else:
                subplot_ax.set_ylim(0, 15000)

            if col_idx == 1:
                subplot_ax.set_yticklabels([])
            subplot_ax.set_xlim(hist_range)
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['left'].set_linewidth(2)
            subplot_ax.spines['bottom'].set_linewidth(2)
            
            # Add model name to leftmost column
            if col_idx == 0:
                subplot_ax.set_ylabel(model_name, fontsize=LABEL_SIZE, rotation=90, 
                                     labelpad=15)
            
            # Add x-label only to bottom row
            if row_idx == 2:
                subplot_ax.set_xlabel('Pixel Value', fontsize=LABEL_SIZE)
            
            # Add legend to top-right subplot
            if row_idx == 0 and col_idx == 1:
                subplot_ax.legend(fontsize=LEGEND_SIZE, loc='upper right')


def plot_power_spectra(ax, dataset='cifar', num_models=2, final_epoch=59):
    """
    Plot power spectra for the second subplot with standard deviation shading.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        num_models: Number of models to use
        final_epoch: Epoch to analyze
    """
    save_path_sae = f'paper_results/{dataset}_sae_final_rfs.npy'
    save_path_pca_init_sae = f'paper_results/{dataset}_pca-ae_final_rfs.npy'
    save_path_dae = f'paper_results/{dataset}_dae_final_rfs.npy'

    try:
        sae_power_spectra = np.load(f'paper_results/{dataset}_sae_power_spectra.npy')
        pca_init_sae_power_spectra = np.load(f'paper_results/{dataset}_pca-init-sae_power_spectra.npy')
        dae_power_spectra = np.load(f'paper_results/{dataset}_dae_power_spectra.npy')
    except:
        print("Computing power spectra...")
        sae_power_spectra, _ = compute_power_spectra(
            save_path_sae, save_path_sae, num_models, final_epoch)
        pca_init_sae_power_spectra, _ = compute_power_spectra(
            save_path_pca_init_sae, save_path_pca_init_sae, num_models, final_epoch)
        dae_power_spectra, _ = compute_power_spectra(
            save_path_dae, save_path_dae, num_models, final_epoch)
        np.save(f'paper_results/{dataset}_sae_power_spectra.npy', sae_power_spectra)
        np.save(f'paper_results/{dataset}_pca-init-sae_power_spectra.npy', pca_init_sae_power_spectra)
        np.save(f'paper_results/{dataset}_dae_power_spectra.npy', dae_power_spectra)
    
    ax.axis('off')
    
    # Grid with 5 columns: AE, PCA-init-SAE, Dev-AE, colorbar, empty space
    grid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax.get_subplotspec(),
                                           width_ratios=[1, 1, 1, 0.05, 0.2], 
                                           wspace=0.2)
    
    ax_sae = plt.subplot(grid[0])
    ax_pca_init_sae = plt.subplot(grid[1])
    ax_dae = plt.subplot(grid[2])
    ax_cbar = plt.subplot(grid[3])
    
    # Average across models
    sae_mean = np.mean(sae_power_spectra, axis=0)
    pca_init_sae_mean = np.mean(pca_init_sae_power_spectra, axis=0)
    dae_mean = np.mean(dae_power_spectra, axis=0)
    
    group_boundaries = [0] + neuron_groups
    
    sae_grouped_means = []
    sae_grouped_stds = []
    pca_init_sae_grouped_means = []
    pca_init_sae_grouped_stds = []
    dae_grouped_means = []
    dae_grouped_stds = []
    group_labels = []
    
    # Process each neuron group
    for i in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[i]
        end_idx = group_boundaries[i+1]
        
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
        
        # SAE
        group_data = sae_mean[start_idx:end_idx]
        group_mean = np.mean(group_data, axis=0)
        group_std = np.std(group_data, axis=0)
        sae_grouped_means.append(group_mean)
        sae_grouped_stds.append(group_std)
        
        # PCA-init-SAE
        group_data = pca_init_sae_mean[start_idx:end_idx]
        group_mean = np.mean(group_data, axis=0)
        group_std = np.std(group_data, axis=0)
        pca_init_sae_grouped_means.append(group_mean)
        pca_init_sae_grouped_stds.append(group_std)
        
        # DAE
        group_data = dae_mean[start_idx:end_idx]
        group_mean = np.mean(group_data, axis=0)
        group_std = np.std(group_data, axis=0)
        dae_grouped_means.append(group_mean)
        dae_grouped_stds.append(group_std)
    
    n_groups = len(group_labels)
    colors = plt.cm.cool(np.linspace(0, 1, n_groups))
    
    # Plot SAE power spectra
    for idx in range(n_groups):
        freq_data = sae_grouped_means[idx][:15]
        std_data = sae_grouped_stds[idx][:15]
        
        ax_sae.plot(freq_data, color=colors[idx], linewidth=2)
        ax_sae.fill_between(
            range(len(freq_data)),
            freq_data - std_data,
            freq_data + std_data,
            color=colors[idx],
            alpha=0.2
        )
    
    # Plot PCA-init-SAE power spectra
    for idx in range(n_groups):
        freq_data = pca_init_sae_grouped_means[idx][:15]
        std_data = pca_init_sae_grouped_stds[idx][:15]
        
        ax_pca_init_sae.plot(freq_data, color=colors[idx], linewidth=2)
        ax_pca_init_sae.fill_between(
            range(len(freq_data)),
            freq_data - std_data,
            freq_data + std_data,
            color=colors[idx],
            alpha=0.2
        )
    
    # Plot DAE power spectra
    for idx in range(n_groups):
        freq_data = dae_grouped_means[idx][:15]
        std_data = dae_grouped_stds[idx][:15]
        
        ax_dae.plot(freq_data, color=colors[idx], linewidth=2)
        ax_dae.fill_between(
            range(len(freq_data)),
            freq_data - std_data,
            freq_data + std_data,
            color=colors[idx],
            alpha=0.2
        )
    
    ylim_top = 30000

    ax_sae.set_xlim(0, 10)
    ax_sae.set_ylim(0, ylim_top)
    ax_sae.set_xticks([0, 5, 10])
    ax_sae.set_xticklabels(['0', '5', '10'], fontsize=TICK_SIZE)
    ax_sae.set_xlabel('Frequency', fontsize=LABEL_SIZE)
    ax_sae.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2)
    ax_sae.set_title('AE', fontsize=TITLE_SIZE, pad=10)
    ax_sae.spines['top'].set_visible(False)
    ax_sae.spines['right'].set_visible(False)
    ax_sae.spines['left'].set_linewidth(2)
    ax_sae.spines['bottom'].set_linewidth(2)

    ax_pca_init_sae.set_xlim(0, 10)
    ax_pca_init_sae.set_ylim(0, ylim_top)
    ax_pca_init_sae.set_xticks([0, 5, 10])
    ax_pca_init_sae.set_xticklabels(['0', '5', '10'], fontsize=TICK_SIZE)
    ax_pca_init_sae.set_xlabel('Frequency', fontsize=LABEL_SIZE)
    ax_pca_init_sae.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2)
    ax_pca_init_sae.set_title('PCA-AE', fontsize=TITLE_SIZE, pad=10)
    ax_pca_init_sae.spines['top'].set_visible(False)
    ax_pca_init_sae.spines['right'].set_visible(False)
    ax_pca_init_sae.spines['left'].set_linewidth(2)
    ax_pca_init_sae.spines['bottom'].set_linewidth(2)

    ax_dae.set_xlim(0, 10)
    ax_dae.set_ylim(0, ylim_top)
    ax_dae.set_xticks([0, 5, 10])
    ax_dae.set_xticklabels(['0', '5', '10'], fontsize=TICK_SIZE)
    ax_dae.set_xlabel('Frequency', fontsize=LABEL_SIZE)
    ax_dae.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2)
    ax_dae.set_title('Dev-AE', fontsize=TITLE_SIZE, pad=10)
    ax_dae.spines['top'].set_visible(False)
    ax_dae.spines['right'].set_visible(False)
    ax_dae.spines['left'].set_linewidth(2)
    ax_dae.spines['bottom'].set_linewidth(2)
    
    # Only show y-axis label on left plot
    ax_sae.set_ylabel('Power', fontsize=LABEL_SIZE)
    ax_pca_init_sae.set_yticks([])
    ax_dae.set_yticks([])
    
    # Colorbar
    colors = plt.cm.cool(np.linspace(0, 1, n_groups))[::-1]
    group_labels = group_labels[::-1]
    discrete_cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, n_groups + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    
    tick_positions = np.arange(n_groups)
    cbar = plt.colorbar(sm, cax=ax_cbar, ticks=tick_positions)
    cbar.set_label("Neuron Groups", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE, width=2)
    cbar.set_ticklabels(group_labels)
    cbar.minorticks_off()
    
    # ax.set_title("Power Spectra of Receptive Fields", fontsize=TITLE_SIZE, pad=40, y=1.05)


def plot_jsd_summary(ax, dataset='cifar', num_models=10):
    """
    Plot summary of JSD values across all models for AE, PCA-AE, and Dev-AE.
    Shows individual model JSDs and mean ± std error bars.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        num_models: Number of models to analyze
    """
    # Load RFs
    sae_rfs = np.load(f'paper_results/{dataset}_sae_final_rfs.npy')
    pca_init_sae_rfs = np.load(f'paper_results/{dataset}_pca-ae_final_rfs.npy')
    dae_rfs = np.load(f'paper_results/{dataset}_dev-ae_final_rfs.npy')
    
    bins = 50
    hist_range = (-20, 20)
    rf_range = slice(0, None)
    
    model_types = [
        ('AE', sae_rfs),
        ('PCA-AE', pca_init_sae_rfs),
        ('Dev-AE', dae_rfs)
    ]
    
    all_jsd_values = []
    all_means = []
    all_stds = []
    model_names = []
    
    # Compute JSD for each model type
    for model_name, model_rfs in model_types:
        jsd_values = []
        
        for model_idx in range(min(num_models, len(model_rfs))):
            model_data = model_rfs[model_idx]
            
            # Transpose if needed
            if dataset.lower() == 'cifar' and model_data.shape[1] == 3:
                model_data = np.transpose(model_data, (0, 2, 3, 1))
                model_data_for_jsd = np.transpose(model_data, (0, 3, 1, 2))
            else:
                model_data_for_jsd = model_data
            
            jsd_mean = compute_rgb_jsd_multi(model_data_for_jsd, rf_range, 
                                            bins=bins, hist_range=hist_range)
            jsd_values.append(jsd_mean)
        
        jsd_values = np.array(jsd_values)
        all_jsd_values.append(jsd_values)
        all_means.append(jsd_values.mean())
        all_stds.append(jsd_values.std())
        model_names.append(model_name)
    
    x_positions = np.arange(len(model_names))
    
    # Define colors for each model type
    model_colors = ['#1a7adb', '#00a65a', '#e82817']  # AE, PCA-AE, Dev-AE
    
    # Scatter individual sub-models
    for i, jsd_vals in enumerate(all_jsd_values):
        ax.scatter(np.full_like(jsd_vals, x_positions[i]), jsd_vals, 
                  alpha=0.6, s=100, color=model_colors[i], label=model_names[i])
    
    # Error bars for mean ± std
    for i in range(len(model_names)):
        ax.errorbar(
            x_positions[i],
            all_means[i],
            yerr=all_stds[i],
            fmt='o',
            color='black',
            ecolor='black',
            capsize=10,
            markersize=10,
            linewidth=2,
            alpha=0.8,
        )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, fontsize=LABEL_SIZE)
    ax.set_ylabel("JSD across RGB", fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


# compute_final_rfs("sae", dataset='cifar', size_ls=None, num_models=10, num_epochs=60)
# compute_final_rfs("pca-ae", dataset='cifar', size_ls=None, num_models=10, num_epochs=60)
# compute_final_rfs("dev-ae", dataset='cifar', size_ls=cifar_size_ls, num_models=10, num_epochs=60)

create_figure_2(dataset='cifar', num_models=10, final_epoch=59)