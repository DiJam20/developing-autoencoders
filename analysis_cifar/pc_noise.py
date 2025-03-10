import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
import torch

from autoencoder import *
from model_utils import *
from solver import *


def add_noise_and_reconstruct(test_images, noise_scale, n_components=128, start_noise_idx=0, end_noise_idx=4):
    # Perform PCA
    pca = PCA(n_components=n_components)
    # print('debug: test_images.shape', test_images.shape,flush=True)
    
    pca_reduced = pca.fit_transform(test_images.reshape(len(test_images), 3*32*32))
    
    # Add different noise to each PCA reduced image to specified PCs
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=(len(pca_reduced), end_noise_idx-start_noise_idx))
    pca_reduced[:, start_noise_idx:end_noise_idx] += noise
    
    # Reconstruct
    return pca.inverse_transform(pca_reduced).reshape(len(test_images), 3, 32, 32)


def get_encoding_diff(model: ConvAutoencoder, original_img: torch.Tensor, noisy_img: torch.Tensor) -> np.ndarray:
    # print('debug: original_img.shape', original_img.shape,flush=True)
    original_img = original_img.unsqueeze(0)
    noisy_img = noisy_img.unsqueeze(0)
    original_encoding = model.encode(original_img)
    noisy_encoding = model.encode(noisy_img)
    return np.abs((original_encoding - noisy_encoding).detach().numpy())


def evaluate_models(test_images, reconstructed_images, sae, dae):
    sae_diffs = []
    dae_diffs = []
    
    for i in range(len(test_images)):
        test_image = test_images[i]
        reconstructed_image = torch.tensor(reconstructed_images[i], dtype=torch.float32)
        
        with torch.no_grad():
            sae_diff = get_encoding_diff(sae, test_image, reconstructed_image)
            dae_diff = get_encoding_diff(dae, test_image, reconstructed_image)

            sae_diffs.append(sae_diff)
            dae_diffs.append(dae_diff)
    
    return np.mean(np.vstack(sae_diffs), axis=0), np.mean(np.vstack(dae_diffs), axis=0)


def plot_neuron_comparison(results, manipulated_neurons, savepath):
    plt.rcParams.update({'font.size': 16})
    
    plt.figure(figsize=(12, 8))
    
    sae_colors = plt.cm.Blues(np.linspace(0.3, 1, len(manipulated_neurons)))
    dae_colors = plt.cm.Reds(np.linspace(0.3, 1, len(manipulated_neurons)))
    
    for i, neuron_pair in enumerate(manipulated_neurons):
        sae_data = np.vstack([run[0] for run in results[neuron_pair]])
        dae_data = np.vstack([run[1] for run in results[neuron_pair]])
        
        # Calculate statistics
        sae_mean = np.mean(sae_data, axis=0)
        sae_std = np.std(sae_data, axis=0)
        dae_mean = np.mean(dae_data, axis=0)
        dae_std = np.std(dae_data, axis=0)
        
        # Plot SAE
        plt.plot(sae_mean, color=sae_colors[i], 
                label=f'SAE (Noisy PCs: {manipulated_neurons[i][0]} - {manipulated_neurons[i][1]-1})', 
                linewidth=2)
        plt.fill_between(range(len(sae_mean)), 
                        sae_mean - sae_std, 
                        sae_mean + sae_std,
                        color=sae_colors[i], alpha=0.1)
        
        # Plot DAE
        plt.plot(dae_mean, color=dae_colors[i], 
                label=f'DAE (Noisy PCs: {manipulated_neurons[i][0]} - {manipulated_neurons[i][1]-1})', 
                linewidth=2)
        plt.fill_between(range(len(dae_mean)),
                        dae_mean - dae_std,
                        dae_mean + dae_std,
                        color=dae_colors[i], alpha=0.1)
    
    plt.title("PC Noise", fontsize=16, pad=20)
    plt.xlabel("Neuron Index", fontsize=16)
    plt.ylabel("Absolute Activation Difference", fontsize=16)
    
    legend = plt.legend(loc='upper right', 
                      fontsize=14,
                      framealpha=0.9,
                      edgecolor='black')
    
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def compute_pc_noise_analysis(num_models, manipulated_neurons):
    result_file = "Results/pc_noise.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    test_images, _ = load_cifar_tensor()

    results = {pair: [] for pair in manipulated_neurons}

    for neuron_pair in tqdm(manipulated_neurons, desc="Processing PC ranges", leave=False):
        # Generate noisy images
        noisy_reconstructed = add_noise_and_reconstruct(test_images, noise_scale=10, start_noise_idx=neuron_pair[0], end_noise_idx=neuron_pair[1])
        
        for iteration in tqdm(range(num_models), desc=f"Testing models for PCs {neuron_pair}", leave=False):
            # modelpath = f'/home/david/mnist_model/'
            # sae = load_model(modelpath+'sae/'+str(iteration), 'sae', 59)
            # dae = load_model(modelpath+'dae/'+str(iteration), 'dae', 59)
            epoch = 49
            run_id = '2025-03-07_13:13:33'
            model_type = 'dae'
            model_path = f"/home/kong/cifar_models/cnn/{run_id}/{model_type}/{iteration}"
            dae = load_conv_model(model_path, model_type=model_type, epoch=epoch)
            model_type = 'sae'
            model_path = f"/home/kong/cifar_models/cnn/{run_id}/{model_type}/{iteration}"
            sae = load_conv_model(model_path, model_type=model_type, epoch=epoch)
            
            # Evaluate models
            sae_diffs, dae_diffs = evaluate_models(test_images, noisy_reconstructed, sae, dae)
            results[neuron_pair].append((sae_diffs, dae_diffs))

    np.save(result_file, results)

def create_ranking_heatmaps(results, manipulated_neurons):
    # First, get a sample to determine the number of neurons
    sample_data = next(iter(results.values()))[0]
    num_neurons = len(sample_data[0])  # Length of SAE diffs for first run
    
    # Initialize arrays to store mean activation differences
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
        
        # Calculate rankings using argsort and flipping so that 1 = highest activation difference
        sae_rankings[:, neuron] = np.argsort(np.argsort(-sae_neuron_diffs)) + 1
        dae_rankings[:, neuron] = np.argsort(np.argsort(-dae_neuron_diffs)) + 1
    
    # Create labels for PC ranges
    pc_labels = [f"({r[0]}-{r[1]-1})" for r in manipulated_neurons]
    
    # Set up the figure
    plt.rcParams.update({'font.size': 14})
    
    xtick_positions = [0, 15, 31, 58, 87, 127]
    xtick_labels = [1, 16, 32, 59, 88, 128]
    # xtick_labels = [1, 16, 32]
    
    blues = plt.cm.Blues_r(np.linspace(0, 1, 7))
    discrete_blues = ListedColormap(blues)
    reds = plt.cm.Reds_r(np.linspace(0, 1, 7))
    discrete_reds = ListedColormap(reds)
    
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, 5)
    
    # SAE Heatmap
    # plt.figure(figsize=(max(12, num_neurons*0.4), 3))
    plt.figure(figsize=(5,3))
    
    ax = sns.heatmap(sae_rankings, annot=False, cmap=discrete_blues, 
                cbar=True, linewidths=.1,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=pc_labels)
    
    ax.set_xticks([p + 0.5 for p in xtick_positions])
    ax.set_xticklabels(xtick_labels)
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([1, 2, 3, 4, 5,6,7])
    cbar.set_ticklabels([1, 2, 3, 4, 5,6,7])
    cbar.minorticks_off()
    cbar.ax.invert_yaxis()
    
    plt.title("SAE: PC Noise Impact Rankings", fontsize=16)
    plt.xlabel("Neuron Index", fontsize=14)
    plt.ylabel("Manipulated PC Range", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Results/pc_noise_heatmap_sae_rankings.png", dpi=300,bbox_inches='tight')
    plt.close()
    
    # DAE Heatmap
    # plt.figure(figsize=(max(12, num_neurons*0.4), 3))
    plt.figure(figsize=(5,3))
    
    ax = sns.heatmap(dae_rankings, annot=False, cmap=discrete_reds, 
                cbar=True,  linewidths=.1,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=pc_labels)
    
    ax.set_xticks([p + 0.5 for p in xtick_positions])
    ax.set_xticklabels(xtick_labels)
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([1, 2, 3, 4, 5,6,7])
    cbar.set_ticklabels([1, 2, 3, 4, 5,6,7])
    cbar.minorticks_off()
    cbar.ax.invert_yaxis()
    
    plt.title("DAE: PC Noise Impact Rankings", fontsize=16)
    plt.xlabel("Neuron Index", fontsize=14)
    plt.ylabel("Manipulated PC Range", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Results/pc_noise_heatmap_dae_rankings.png", dpi=300)
    plt.close()
    
    return sae_rankings, dae_rankings


def analyze_and_visualize_pc_noise():
    # Load the results
    results_file = "Results/pc_noise.npy"
    results = np.load(results_file, allow_pickle=True).item()
    
    manipulated_neurons = [(0, 5), (6, 10), (10, 15), (16, 27), (28, 47), (48, 89), (90, 127)]
    
    sae_rankings, dae_rankings = create_ranking_heatmaps(results, manipulated_neurons)
        
    return sae_rankings, dae_rankings

def run_pc_noise_analysis(num_models):
    manipulated_neurons = [(0, 5), (6, 10), (10, 15), (16, 27), (28, 47), (48, 89), (90, 127)]
    compute_pc_noise_analysis(num_models, manipulated_neurons)
    results = np.load("Results/pc_noise.npy", allow_pickle=True).item()

    plot_neuron_comparison(results, manipulated_neurons, "Results/pc_noise.png")

    analyze_and_visualize_pc_noise()