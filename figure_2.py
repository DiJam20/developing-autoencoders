import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap

from rf_computation import compute_rfs
from power_spectra import compute_power_spectra
from rf_stability import compute_angles_between_rfs, compute_average_angles_matrix

LABEL_SIZE = 22
TICK_SIZE = 22
LEGEND_SIZE = 16
TITLE_SIZE = 26

MODEL_IDX_FOR_EXAMPLE_RF = 1

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

cifar_size_ls = [6, 6, 6, 6, 6, 6, 
                10, 10, 10, 10, 10, 10,
                16, 16, 16, 16, 16, 16,
                28, 28, 28, 28, 28, 28,
                48, 48, 48, 48, 48, 48, 48, 48, 48,
                90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                128, 128, 128, 128, 128, 128, 128, 128, 
                128, 128, 128, 128, 128, 128, 128, 128, 128
                ]

# Neuron group endpoints
neuron_groups = [6, 12, 18, 28, 48, 90, 128]

os.makedirs("Results/figures/png", exist_ok=True)
os.makedirs("Results/figures/svg", exist_ok=True)


def create_figure_2(dataset='cifar', num_models=2, final_epoch=59):
    """
    Create Figure 2 with three subplots:
    1. Example RFs (top row)
    2. Power spectra (bottom left)
    3. RF stability (bottom right)
    
    Args:
        dataset: Dataset name ('cifar')
        num_models: Number of models to use
        final_epoch: Final epoch to use for analysis
    """
    fig = plt.figure(figsize=(20, 12))
    
    gs = gridspec.GridSpec(2, 2,  width_ratios=[1, 1], height_ratios=[2.2, 1.2], wspace=0.5, hspace=0.3)
    
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :])
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 0])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1])
    
    ax_top = plt.subplot(gs_top[0])
    ax_bottom_left = plt.subplot(gs_bottom_left[0])
    ax_bottom_right = plt.subplot(gs_bottom_right[0])
    
    plot_example_rfs(ax_top, dataset, final_epoch)
    plot_power_spectra(ax_bottom_left, dataset, num_models, final_epoch)
    plot_rf_stability(ax_bottom_right, dataset, num_models, final_epoch)
    
    # Add figure labels
    ax_top.text(-0.08, 1, 'A', transform=ax_top.transAxes, 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom_left.text(-0.2, 1.2275, 'B', transform=ax_bottom_left.transAxes, 
                        fontsize=TITLE_SIZE, fontweight='bold')
    ax_bottom_right.text(-0.2, 1.2275, 'C', transform=ax_bottom_right.transAxes, 
                         fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(f"Results/figures/png/{dataset}_figure_2.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_figure_2.svg", bbox_inches='tight')
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
    sae_rfs = np.load(f'Results/{dataset}_sae_rfs.npy')
    dae_rfs = np.load(f'Results/{dataset}_dae_rfs.npy')
    
    # Get the neuron indices for each group
    group_boundaries = [0] + neuron_groups
    
    ax.axis('off')
    
    num_groups = len(group_boundaries) - 1
    
    # Grid with 3 rows: AE, Dev-AE, arrow
    outer_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=ax.get_subplotspec(),
                                                 height_ratios=[1, 1, 0.15], 
                                                 hspace=0.1)
    
    grid_wspace = 0.15
    rf_wspace = 0.1
    rf_hspace = 0
    
    # Grids for AE and Dev-AE different neuron groups
    ae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                     subplot_spec=outer_grid[0],
                                                     wspace=grid_wspace)
    devae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                        subplot_spec=outer_grid[1],
                                                        wspace=grid_wspace)
    
    arrow_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[2])
    
    group_labels = []
    for i in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[i]
        end_idx = group_boundaries[i+1]
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
    
    row_labels = ["AE", "Dev-AE"]
    
    # Separate axis for the AE and Dev-AE labels
    ae_label_ax = plt.subplot(outer_grid[0])
    ae_label_ax.axis('off')
    ae_label_ax.text(-0.05, 0.5, row_labels[0], rotation=90, 
                     transform=ae_label_ax.transAxes, 
                     ha='center', va='center', fontsize=LABEL_SIZE)
    devae_label_ax = plt.subplot(outer_grid[1])
    devae_label_ax.axis('off')
    devae_label_ax.text(-0.05, 0.5, row_labels[1], rotation=90, 
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
            if neuron_idx < len(sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx]):
                rf_data = sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx, neuron_idx]
                
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
            
            if neuron_idx < len(dae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx]):
                rf_data = dae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx, neuron_idx]
                
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


def plot_power_spectra(ax, dataset='cifar', num_models=2, final_epoch=59):
    """
    Plot power spectra for the second subplot with standard deviation shading.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        num_models: Number of models to use
        final_epoch: Epoch to analyze
    """
    save_path_sae = f'Results/{dataset}_sae_rfs.npy'
    save_path_dae = f'Results/{dataset}_dae_rfs.npy'
    
    try:
        sae_power_spectra = np.load(f'Results/{dataset}_sae_power_spectra.npy')
        dae_power_spectra = np.load(f'Results/{dataset}_dae_power_spectra.npy')
    except:
        print("Computing power spectra...")
        sae_power_spectra, dae_power_spectra = compute_power_spectra(
            save_path_sae, save_path_dae, num_models, final_epoch)
        np.save(f'Results/{dataset}_sae_power_spectra.npy', sae_power_spectra)
        np.save(f'Results/{dataset}_dae_power_spectra.npy', dae_power_spectra)
    
    ax.axis('off')
    
    # Grid with 3 columns: AE, Dev-AE, colorbar
    grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                           width_ratios=[1, 1, 0.05], 
                                           wspace=0.2)
    
    ax_sae = plt.subplot(grid[0])
    ax_dae = plt.subplot(grid[1])
    ax_cbar = plt.subplot(grid[2])
    
    # Average across models
    sae_mean = np.mean(sae_power_spectra, axis=0)
    dae_mean = np.mean(dae_power_spectra, axis=0)
    
    group_boundaries = [0] + neuron_groups
    
    sae_grouped_means = []
    sae_grouped_stds = []
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
    ax_sae.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax_sae.set_title('AE', fontsize=TITLE_SIZE, pad=10)
    ax_sae.spines['top'].set_visible(False)
    ax_sae.spines['right'].set_visible(False)

    ax_dae.set_xlim(0, 10)
    ax_dae.set_ylim(0, ylim_top)
    ax_dae.set_xticks([0, 5, 10])
    ax_dae.set_xticklabels(['0', '5', '10'], fontsize=TICK_SIZE)
    ax_dae.set_xlabel('Frequency', fontsize=LABEL_SIZE)
    ax_dae.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax_dae.set_title('Dev-AE', fontsize=TITLE_SIZE, pad=10)
    ax_dae.spines['top'].set_visible(False)
    ax_dae.spines['right'].set_visible(False)
    
    # Only show y-axis label on left plot
    ax_sae.set_ylabel('Power', fontsize=LABEL_SIZE)
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
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_ticklabels(group_labels)
    cbar.minorticks_off()
    
    # ax.set_title("Power Spectra of Receptive Fields", fontsize=TITLE_SIZE, pad=40, y=1.05)


def plot_rf_stability(ax, dataset='cifar', num_models=2, final_epoch=59):
    """
    Plot RF stability heatmap for the third subplot.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        num_models: Number of models to use
        final_epoch: Final epoch to use
    """
    ax.axis('off')
    
    # Grid with 3 columns: AE, Dev-AE, colorbar
    grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                           width_ratios=[1, 1, 0.05],
                                           wspace=0.2)
    
    ax_sae = plt.subplot(grid[0])
    ax_dae = plt.subplot(grid[1])
    ax_cbar = plt.subplot(grid[2])
    
    # Set the same aspect ratio for both heatmap axes
    ax_sae.set_aspect('auto')
    ax_dae.set_aspect('auto')
    
    # Compute angles between RFs if not already computed
    compare_final_epoch = True
    for model_type in ['sae', 'dae']:
        comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
        angles_file = f"Results/{dataset}_{model_type}_rf_stability_{comparison_type}_angles.npy"
        
        if not os.path.exists(angles_file):
            print(f"Computing angles for {model_type}...")
            compute_angles_between_rfs(model_type, dataset, compare_final_epoch, num_models, final_epoch+1)
    
    # Get angle matrices
    sae_angles, _ = compute_average_angles_matrix('sae', dataset, compare_final_epoch)
    dae_angles, dae_non_computable = compute_average_angles_matrix('dae', dataset, compare_final_epoch)
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 90)
    
    # SAE heatmap
    sns.heatmap(
        sae_angles[:, :],
        cmap=cmap,
        vmin=0,
        vmax=90,
        cbar=False,
        ax=ax_sae
    )
    
    # DAE heatmap
    sns.heatmap(
        dae_angles[:, :],
        cmap=cmap,
        vmin=0,
        vmax=90,
        cbar=False,
        ax=ax_dae
    )
    
    # Add gray to non-computable areas in DAE heatmap
    if dae_non_computable is not None:
        cmap_grey = ListedColormap(['grey'])
        sns.heatmap(
            dae_non_computable[:, :],
            cmap=cmap_grey,
            cbar=False,
            alpha=1,
            ax=ax_dae
        )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0", "45", "90"], fontsize=TICK_SIZE)
    cbar.set_label("Cosine Angle Difference (°)", fontsize=LABEL_SIZE, labelpad=10)
    cbar.minorticks_off()
    
    ax_sae.set_title("AE", fontsize=TITLE_SIZE, pad=10)
    ax_dae.set_title("Dev-AE", fontsize=TITLE_SIZE, pad=10)
    
    max_epochs = sae_angles.shape[1]
    mid_epoch = max_epochs // 2
    for a in [ax_sae, ax_dae]:
        a.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
        a.set_xticklabels(["1", f"{mid_epoch}", f"{max_epochs}"], 
                         fontsize=TICK_SIZE, rotation=0)
        a.set_xlabel("Epochs", fontsize=LABEL_SIZE)
    
    num_pcs = sae_angles.shape[0]
    mid_pc = num_pcs // 2
    ax_sae.set_yticks([0.5, mid_pc - 0.5, num_pcs - 0.5])
    ax_sae.set_yticklabels(["1", f"{mid_pc}", f"{num_pcs}"], 
                           fontsize=TICK_SIZE, rotation=0)
    ax_sae.set_ylabel("Neuron Index", fontsize=LABEL_SIZE)
    
    ax_dae.set_yticks([])
    ax_dae.set_yticklabels([])
    ax_dae.set_ylabel("")
    
    # ax.set_title("Receptive Field Stability Over Training", fontsize=TITLE_SIZE, pad=40, y=1.05)


compute_rfs("sae", dataset='cifar', size_ls=None, num_models=2, num_epochs=60)
compute_rfs("dae", dataset='cifar', size_ls=cifar_size_ls, num_models=2, num_epochs=60)

create_figure_2(dataset='cifar', num_models=2, final_epoch=59)