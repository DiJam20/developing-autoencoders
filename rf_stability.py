import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import act_max_util as amu

from autoencoder import *
from solver import *
from model_utils import *



IMG_WIDTH, IMG_HEIGHT = 28, 28
MAX_NEURONS = 32

steps = 100               # perform 100 iterations
alpha = torch.tensor(100) # learning rate (step size)
verbose = False           # print activation every step
L2_Decay = True           # enable L2 decay regularizer
Gaussian_Blur = False     # enable Gaussian regularizer
Norm_Crop = False         # enable norm regularizer
Contrib_Crop = False      # enable contribution regularizer


def compute_rf_for_single_model(model_idx, model_type, size_ls=None, num_epochs=60):
    rf_matrix = []
    
    for epoch in range(num_epochs):
        data = torch.randn(IMG_WIDTH, IMG_HEIGHT)
        data = data.unsqueeze(0)
        input = data.view(data.size(0), -1)
        input.requires_grad_(True)

        ae = load_model(f'/home/david/mnist_model/{model_type}/{model_idx}', model_type, epoch)
        
        layer_name = 'bottle_neck'
        activation_dictionary = {}

        ae.encoder.encoder_3.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

        loop_size = size_ls[epoch] if model_type == "dae" else MAX_NEURONS

        rf_ls = []
        for i in range(loop_size):
            output = amu.act_max(network=ae,
                            input=input,
                            layer_activation=activation_dictionary,
                            layer_name=layer_name,
                            unit=i,
                            steps=steps,
                            alpha=alpha,
                            verbose=verbose,
                            L2_Decay=L2_Decay,
                            Gaussian_Blur=Gaussian_Blur,
                            Norm_Crop=Norm_Crop,
                            Contrib_Crop=Contrib_Crop,
                            )
            rf_ls.append(output.detach().numpy())
        
        max_size = MAX_NEURONS

        # Pad the rf_ls list to max_size before converting to numpy array
        if loop_size < max_size:
            # Create a zero RF with the same shape as the first RF
            zero_rf = np.zeros_like(rf_ls[0])
            
            # Add zero RFs until we reach max_size
            for _ in range(loop_size, max_size):
                rf_ls.append(zero_rf.copy())
        
        # Convert to numpy array and apply squeeze as in the original function
        rf_ls = np.array(rf_ls).squeeze()
        rf_matrix.append(rf_ls)

    return rf_matrix


def compute_rfs(model_type, size_ls=None, num_models=10, num_epochs=60):

    result_file = f"Results/{model_type}_rfs.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file)

    # For all different models
    rf_matrices = []

    for model_idx in tqdm(range(num_models), desc=f"Model {model_type}"):
        rf_matrix = compute_rf_for_single_model(
            model_idx, 
            model_type, 
            size_ls=size_ls,
            num_epochs=num_epochs
        )

        rf_matrices.append(rf_matrix)

    np.save(result_file, rf_matrices)

    return rf_matrices


def display_ready_rf(rf):
    if len(rf.shape) == 1:
        return rf.reshape(IMG_WIDTH, IMG_HEIGHT)
    return rf


def plot_rfs_for_single_model(model_type, model_idx=0, epoch_idx=-1):
    rf_matrices = np.load(f"Results/{model_type}_rf_stability_all_rfs.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
    rf_ls = rf_matrix[epoch_idx]
    
    fig = plt.figure(figsize=(20,9))
    fig.suptitle(f"Receptive Fields ({model_type})", fontsize=24)
    
    for i in range(MAX_NEURONS):
        plt.subplot(4,8,i+1)
        plt.title(str(i+1))
        plt.axis('off')
        
        # Only plot if the index exists in rf_ls, otherwise leave empty
        if i < len(rf_ls):
            plt.imshow(display_ready_rf(rf_ls[i]), cmap='gray')
        else:
            plt.imshow(np.zeros((28,28)), cmap='gray')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"Results/{model_type}_rf_stability_model_{model_idx}_epoch_{epoch_idx}.png", dpi=300)
    plt.savefig(f"Results/{model_type}_rf_stability_model_{model_idx}_epoch_{epoch_idx}.svg")


def plot_rf_over_time(model_type, model_idx, neuron_idx, num_epochs=60):
    rf_matrices = np.load(f"Results/{model_type}_rf_stability_all_angles.npy", allow_pickle=True)
    rf_matrix = rf_matrices[model_idx]
    
    fig, axes = plt.subplots(10, 5, figsize=(10, 20))
    axes = axes.ravel()

    for i in range(num_epochs):
        axes[i].imshow(display_ready_rf(rf_matrix[i][neuron_idx]), cmap='gray')
        axes[i].axis('off')

    fig.suptitle(f"Receptive Field Development of Neuron 0 ({model_type})", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def compute_angles_between_rfs(model_type, num_models, num_epochs, num_neurons):
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


def compute_average_angles_matrix(model_type):
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


def create_heatmap(model_type):
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


def analyze_rf_stability(model_type, size_ls=None, num_models=10, num_epochs=60):
    compute_rfs(model_type, size_ls=size_ls, num_models=num_models, num_epochs=num_epochs)
    compute_angles_between_rfs(model_type, num_models, num_epochs, MAX_NEURONS)
    create_heatmap(model_type)
    plot_rfs_for_single_model(model_type, model_idx=0, epoch_idx=-1)
    plot_rf_over_time(model_type, model_idx=0, neuron_idx=0, num_epochs=num_epochs)
