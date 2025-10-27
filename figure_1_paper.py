import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
from autoencoder import *
from model_utils import *
from solver import *
from validation_loss import get_train_loss_per_epoch, get_vali_loss_per_epoch
from dimensionality_development import compute_dimensionality_matrix

LABEL_SIZE = 22
TICK_SIZE = 22
LEGEND_SIZE = 20
TITLE_SIZE = 26

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

# CHANGE THESE PARAMETERS
base_path = '/home/david/'
num_models = 40
num_epochs = 60
size_ls = cifar_size_ls
dataset = 'cifar'

os.makedirs("paper_results/figures/png", exist_ok=True)
os.makedirs("paper_results/figures/pdf", exist_ok=True)


# PLOT LOSS CURVES
def plot_accuracy_over_epochs(
        sae_train_loss, 
        sae_vali_loss, 
        pca_init_sae_train_loss, 
        pca_init_sae_vali_loss,
        dae_train_loss, 
        dae_vali_loss, 
        ax,
        ):
    """
    Plot both training and validation loss curves for SAE and DAE models.
    
    Args:
        sae_train_loss: Training loss data for SAE models
        dae_train_loss: Training loss data for DAE models
        sae_vali_loss: Validation loss data for SAE models
        dae_vali_loss: Validation loss data for DAE models
        ax: Matplotlib axis to plot on
    """
    # Calculate means and standard deviations
    sae_train_mean = np.mean(sae_train_loss, axis=0)
    sae_train_std = np.std(sae_train_loss, axis=0)

    pca_init_sae_train_mean = np.mean(pca_init_sae_train_loss, axis=0)
    pca_init_sae_train_std = np.std(pca_init_sae_train_loss, axis=0)

    dae_train_mean = np.mean(dae_train_loss, axis=0)
    dae_train_std = np.std(dae_train_loss, axis=0)
    
    sae_vali_mean = np.mean(sae_vali_loss, axis=0)
    sae_vali_std = np.std(sae_vali_loss, axis=0)

    pca_init_sae_vali_mean = np.mean(pca_init_sae_vali_loss, axis=0)
    pca_init_sae_vali_std = np.std(pca_init_sae_vali_loss, axis=0)

    dae_vali_mean = np.mean(dae_vali_loss, axis=0)
    dae_vali_std = np.std(dae_vali_loss, axis=0)
    
    # Training loss
    sae_train_line, = ax.plot(sae_train_mean, color='#1a7adb', linewidth=2)
    ax.fill_between(range(len(sae_train_loss[0])),
                    sae_train_mean - sae_train_std,
                    sae_train_mean + sae_train_std,
                    color='#1a7adb', alpha=0.2)

    pca_init_sae_train_line, = ax.plot(pca_init_sae_train_mean, color='#00a65a', linewidth=2)
    ax.fill_between(range(len(pca_init_sae_train_loss[0])),
                    pca_init_sae_train_mean - pca_init_sae_train_std,
                    pca_init_sae_train_mean + pca_init_sae_train_std,
                    color='#00a65a', alpha=0.2)

    dae_train_line, = ax.plot(dae_train_mean, color='#e82817', linewidth=2)
    ax.fill_between(range(len(dae_train_loss[0])),
                    dae_train_mean - dae_train_std,
                    dae_train_mean + dae_train_std,
                    color='#e82817', alpha=0.2)

    # Validation loss
    sae_vali_line, = ax.plot(sae_vali_mean, color='#1a7adb', linewidth=2, linestyle='--')
    ax.fill_between(range(len(sae_vali_loss[0])),
                    sae_vali_mean - sae_vali_std,
                    sae_vali_mean + sae_vali_std,
                    color='#1a7adb', alpha=0.2)

    pca_init_sae_vali_line, = ax.plot(pca_init_sae_vali_mean, color='#00a65a', linewidth=2, linestyle='--')
    ax.fill_between(range(len(pca_init_sae_vali_loss[0])),
                    pca_init_sae_vali_mean - pca_init_sae_vali_std,
                    pca_init_sae_vali_mean + pca_init_sae_vali_std,
                    color='#00a65a', alpha=0.2)

    dae_vali_line, = ax.plot(dae_vali_mean, color='#e82817', linewidth=2, linestyle='--')
    ax.fill_between(range(len(dae_vali_loss[0])),
                    dae_vali_mean - dae_vali_std,
                    dae_vali_mean + dae_vali_std,
                    color='#e82817', alpha=0.2)
    
    ax.set_xticks([0, 29, 59])
    ax.set_xticklabels([1, 30, 60])
    ax.set_yticks([0, 0.1])
    ax.set_xlabel('Epochs', fontsize=LABEL_SIZE)
    ax.set_ylabel('MSE Loss', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Train'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Val'),
        Line2D([0], [0], color='#1a7adb', marker='o', linestyle='None', markersize=10, label='AE'),
        Line2D([0], [0], color='#00a65a', marker='o', linestyle='None', markersize=10, label='PCA-AE'),
        Line2D([0], [0], color='#e82817', marker='o', linestyle='None', markersize=10, label='Dev-AE'),
    ]

    ax.legend(handles=legend_elements, 
              loc='upper right', 
              fontsize=LEGEND_SIZE, 
              ncol=1)


# PLOT DIMENSIONALITY DEVELOPMENT
def plot_dimensionality_comparison(dataset, ax):
    """
    Plot dimensionality development for SAE and DAE models.
    
    Args:
        dataset: Dataset name (e.g., 'cifar')
        ax: Matplotlib axis to plot on
    """
    sae_file = f"paper_results/{dataset}_sae_dimensionality.npy"
    pca_init_sae_file = f"paper_results/{dataset}_pca-ae_dimensionality.npy"
    dae_file = f"paper_results/{dataset}_dev-ae_dimensionality.npy"
    if not os.path.exists(sae_file) or not os.path.exists(dae_file):
        print(f"Results files not found.")
        return None, None, None, None

    sae_results = np.load(sae_file, allow_pickle=True)
    pca_init_sae_results = np.load(pca_init_sae_file, allow_pickle=True)
    dae_results = np.load(dae_file, allow_pickle=True)

    # Calculate mean and standard deviation across models
    sae_mean = np.mean(sae_results, axis=0)
    sae_std = np.std(sae_results, axis=0)
    pca_init_sae_mean = np.mean(pca_init_sae_results, axis=0)
    pca_init_sae_std = np.std(pca_init_sae_results, axis=0)
    dae_mean = np.mean(dae_results, axis=0)
    dae_std = np.std(dae_results, axis=0)
        
    epochs = np.arange(1, len(sae_mean) + 1)

    ax.plot(epochs, sae_mean, label='AE', color='#1a7adb', linewidth=2)
    ax.fill_between(epochs, sae_mean - sae_std, sae_mean + sae_std, color='#1a7adb', alpha=0.2)
    ax.plot(epochs, pca_init_sae_mean, label='PCA-AE', color='#00a65a', linewidth=2)
    ax.fill_between(epochs, pca_init_sae_mean - pca_init_sae_std, pca_init_sae_mean + pca_init_sae_std, color='#00a65a', alpha=0.2)
    ax.plot(epochs, dae_mean, label='Dev-AE', color='#e82817', linewidth=2)
    ax.fill_between(epochs, dae_mean - dae_std, dae_mean + dae_std, color='#e82817', alpha=0.2)

    ax.set_xticks([0, 29, 59])
    ax.set_xticklabels([1, 30, 60])
    ax.set_xlabel('Epochs', fontsize=LABEL_SIZE)
    ax.set_ylabel('Intrinsic Dimensionality', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend(fontsize=LEGEND_SIZE)
    # ax.set_title('Dimensionality', fontsize=TITLE_SIZE, pad=20)


# PLOT RECONSTRUCTIONS
def plot_reconstructions(dataset, base_path, iteration, num_examples, ax):
    """
    Plot original images and their reconstructions using SAE and DAE models.
    
    Args:
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
        iteration: Model iteration/index to use
        num_examples: Number of examples to show
        ax: Matplotlib axis to plot on
    """    
    # Load test images
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_list()
        images = test_images[:num_examples]
        model_path = f'{base_path}mnist_models/'
        sae = load_model(f"{model_path}sae/{iteration}", 'sae', 59)
        dae = load_model(f"{model_path}dae/{iteration}", 'dae', 59)
        images_tensor = torch.tensor(images, dtype=torch.float32)
        with torch.no_grad():
            flattened = images_tensor.view(images_tensor.size(0), -1)
            _, sae_decoded = sae(flattened)
            _, dae_decoded = dae(flattened)
            sae_reconstructions = sae_decoded.view(num_examples, 28, 28).cpu().numpy()
            dae_reconstructions = dae_decoded.view(num_examples, 28, 28).cpu().numpy()
    else:
        test_images, _ = load_cifar_list()
        images = test_images[:num_examples]
        model_path = f"{base_path}cifar_models/"
        sae = load_conv_model(f"{model_path}sae/{iteration}", 'sae', 59)
        pca_init_sae = load_conv_model(f"{model_path}pca-ae/{iteration}", 'pca-ae', 59)
        dae = load_conv_model(f"{model_path}dae/{iteration}", 'dae', 59)
        images_tensor = torch.tensor(images, dtype=torch.float32)
        with torch.no_grad():
            _, sae_decoded = sae(images_tensor)
            _, pca_init_sae_decoded = pca_init_sae(images_tensor)
            _, dae_decoded = dae(images_tensor)
            sae_reconstructions = sae_decoded.view(num_examples, 3, 32, 32).cpu().numpy()
            pca_init_sae_reconstructions = pca_init_sae_decoded.view(num_examples, 3, 32, 32).cpu().numpy()
            dae_reconstructions = dae_decoded.view(num_examples, 3, 32, 32).cpu().numpy()
        # Transpose images from (N, C, H, W) to (N, H, W, C) for plotting
        images = np.transpose(images, (0, 2, 3, 1))
        sae_reconstructions = np.transpose(sae_reconstructions, (0, 2, 3, 1))
        pca_init_sae_reconstructions = np.transpose(pca_init_sae_reconstructions, (0, 2, 3, 1))
        dae_reconstructions = np.transpose(dae_reconstructions, (0, 2, 3, 1))
    
    grid = gridspec.GridSpecFromSubplotSpec(4, num_examples, subplot_spec=ax.get_subplotspec(), wspace=-0.1, hspace=0.05)

    row_labels = ["Original", "AE", "PCA-AE", "Dev-AE"]

    for row in range(4):
        for col in range(num_examples):
            curr_ax = plt.subplot(grid[row, col])
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            if row == 0:  # Original
                img_to_show = images[col]
            elif row == 1:  # SAE
                img_to_show = sae_reconstructions[col]
            elif row == 2:  # PCA-AE
                img_to_show = pca_init_sae_reconstructions[col]
            else:  # DAE
                img_to_show = dae_reconstructions[col]
            
            if dataset.lower() == "mnist":
                curr_ax.imshow(img_to_show, cmap='gray')
            else:
                img_to_show = np.clip(img_to_show, 0, 1)
                curr_ax.imshow(img_to_show)
            
            # Row labels
            if col == 0:
                curr_ax.text(-0.7, 0.5, row_labels[row], rotation=45, 
                            transform=curr_ax.transAxes, ha='center', va='center',
                            fontsize=LABEL_SIZE)
    
    # ax.set_title('Reconstructions', fontsize=TITLE_SIZE, pad=20)
    ax.axis('off')


# CREATE COMBINED FIGURE
def create_combined_figure(
        sae_train_loss, 
        sae_vali_loss, 
        pca_init_sae_train_loss, 
        pca_init_sae_vali_loss,
        dae_train_loss, 
        dae_vali_loss, 
        dataset, 
        base_path,
        ):
    """
    Create a single figure that combines loss curves, dimensionality development,
    and reconstruction examples for CIFAR dataset.
    
    Args:
        sae_train_loss: Training loss data for SAE models
        dae_train_loss: Training loss data for DAE models
        sae_vali_loss: Validation loss data for SAE models
        dae_vali_loss: Validation loss data for DAE models
        dataset: Dataset name (e.g., 'cifar')
        base_path: Base path to model directories
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(15, 5))
    outer_grid = gridspec.GridSpec(1, 3, wspace=0.4)
    
    # Label positioning
    label_x, label_y = -0.18, 1.07
    
    # Column 1: Loss Curves
    ax1 = plt.subplot(outer_grid[0])
    plot_accuracy_over_epochs(
        sae_train_loss, 
        sae_vali_loss, 
        pca_init_sae_train_loss, 
        pca_init_sae_vali_loss,
        dae_train_loss, 
        dae_vali_loss, 
        ax1
        )
    ax1.text(label_x, label_y, 'A', transform=ax1.transAxes, 
             fontsize=TITLE_SIZE, fontweight='bold')
    
    # Column 2: Dimensionality Development
    ax2 = plt.subplot(outer_grid[1])
    plot_dimensionality_comparison(dataset, ax2)
    ax2.text(label_x, label_y, 'B', transform=ax2.transAxes, 
             fontsize=TITLE_SIZE, fontweight='bold')
    
    # Column 3: Reconstruction Examples
    ax3 = plt.subplot(outer_grid[2])
    plot_reconstructions(dataset, base_path, iteration=0, num_examples=3, ax=ax3)
    ax3.text(label_x, label_y, 'C', transform=ax3.transAxes, 
             fontsize=TITLE_SIZE, fontweight='bold')
        
    plt.savefig(f"paper_results/figures/png/figure_1.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"paper_results/figures/pdf/figure_1.pdf", bbox_inches='tight')
    plt.close()


if dataset == "mnist":
    sae_train_loss = get_train_loss_per_epoch('sae', 'mnist')
    dae_train_loss = get_train_loss_per_epoch('dae', 'mnist')
    sae_vali_loss = get_vali_loss_per_epoch('sae', 'mnist')
    dae_vali_loss = get_vali_loss_per_epoch('dae', 'mnist')
else:
    sae_train_loss = get_train_loss_per_epoch('sae', 'cifar', 10)
    pca_init_train_loss = get_train_loss_per_epoch('pca-ae', 'cifar', num_models)
    dae_train_loss = get_train_loss_per_epoch('dev-ae', 'cifar', num_models)

    sae_vali_loss = get_vali_loss_per_epoch('sae', 'cifar', 10)
    pca_init_vali_loss = get_vali_loss_per_epoch('pca-ae', 'cifar', num_models)
    dae_vali_loss = get_vali_loss_per_epoch('dev-ae', 'cifar', num_models)

# compute_dimensionality_matrix('sae', dataset, size_ls, 10, num_epochs, base_path)
# compute_dimensionality_matrix('pca-ae', dataset, size_ls, 10, num_epochs, base_path)
# compute_dimensionality_matrix('dev-ae', dataset, size_ls, 3, num_epochs, base_path)

create_combined_figure(
    sae_train_loss, 
    sae_vali_loss, 
    pca_init_train_loss, 
    pca_init_vali_loss,
    dae_train_loss, 
    dae_vali_loss, 
    dataset, 
    base_path)