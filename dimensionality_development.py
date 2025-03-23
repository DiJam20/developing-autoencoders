import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model_utils import *

def twonn_dimension(activations):    
    # Find 3 nearest neighbors (point itself + 2 neighbors)
    nbrs = NearestNeighbors(n_neighbors=3).fit(activations)
    distances, _ = nbrs.kneighbors(activations)
    
    # Get ratios of distances to 1st and 2nd neighbors
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = r2 / r1

    dimensionality = 1.0 / np.mean(np.log(mu))
    return dimensionality
    

def participation_ratio(activation_matrix):
    # Calculate covariance matrix and participation ratio directly on bottleneck activations
    cov_matrix = np.cov(activation_matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.abs(eigenvalues)
    
    # Apply participation ratio formula to eigenvalues
    if np.sum(eigenvalues) > 0:
        dimensionality = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
    else:
        dimensionality = 0
        
    return dimensionality

def calculate_dimensionality_for_single_model(model_idx, model_type, test_images, dataset, size_ls, num_epochs, base_path):
    if dataset.lower() == 'mnist':
        base_path = f"{base_path}mnist_models/{model_type}/{model_idx}"
        loader_function = load_model
    elif dataset.lower() == 'cifar':
        base_path = f"{base_path}cifar_models/{model_type}/{model_idx}"
        loader_function = load_conv_model
    
    dimensionality_over_time = []
    
    # Calculate dimensionality for each epoch
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        # Load model for this epoch
        ae = loader_function(base_path, model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        # Get bottleneck activations for all test images
        all_activations = []
        for image in test_images:
            if dataset.lower() == 'mnist':
                image = torch.tensor(image, dtype=torch.float32).reshape(-1)
                with torch.no_grad():
                    encoded = ae.encode(image)
                    all_activations.append(encoded.detach().numpy())
            else:
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    encoded = ae.encode(image)
                    encoded = torch.squeeze(encoded)
                    all_activations.append(encoded.detach().numpy())
        
        # Stack all activations into a single matrix [n_samples, n_features]
        activations_matrix = np.stack(all_activations)

        # Calculate dimensionality using twonn
        twonn_dimensionality = twonn_dimension(activations_matrix)
        dimensionality_over_time.append(twonn_dimensionality)
    
    return dimensionality_over_time


def compute_dimensionality_matrix(model_type, dataset='mnist', size_ls=None, num_models=10, num_epochs=60, base_path='/home/david/'):
    result_file = f"Results/{dataset}_{model_type}_dimensionality.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    if dataset.lower() == 'mnist':
        test_images, _ = load_mnist_list()
    elif dataset.lower() == 'cifar':
        test_images, _ = load_cifar_list()
    
    all_angles = []
    for idx in range(num_models):
        result = calculate_dimensionality_for_single_model(idx, model_type, test_images, dataset, size_ls, num_epochs, base_path)
        all_angles.append(result)

    np.save(result_file, all_angles)
    print(f"Results saved to {result_file}")


def plot_dimensionality_comparison(dataset):
    sae_file = f"Results/{dataset}_sae_dimensionality.npy"
    dae_file = f"Results/{dataset}_dae_dimensionality.npy"
    if not os.path.exists(sae_file) or not os.path.exists(dae_file):
        print(f"Results files not found.")
        return

    sae_results = np.load(sae_file, allow_pickle=True)
    dae_results = np.load(dae_file, allow_pickle=True)

    # Calculate mean and standard deviation across models
    sae_mean = np.mean(sae_results, axis=0)
    sae_std = np.std(sae_results, axis=0)
    dae_mean = np.mean(dae_results, axis=0)
    dae_std = np.std(dae_results, axis=0)

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(sae_mean) + 1)

    plt.plot(epochs, sae_mean, label='AE', color='#1a7adb', linewidth=2)
    plt.fill_between(epochs, sae_mean - sae_std, sae_mean + sae_std, color='#1a7adb', alpha=0.2)
    plt.plot(epochs, dae_mean, label='DevAE', color='#e82817', linewidth=2)
    plt.fill_between(epochs, dae_mean - dae_std, dae_mean + dae_std, color='#e82817', alpha=0.2)

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Intrinsic Dimensionality', fontsize=18)
    plt.title(f'Bottleneck Dimensionality During Training ({dataset.upper()})', fontsize=20, pad=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc='lower right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(f"Results/figures/png/{dataset}_dimensionality_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_dimensionality_comparison.svg", bbox_inches='tight')
    plt.close()


def run_dimensionality_analysis(dataset, size_ls, num_models, num_epochs, base_path):
    print(f"Running dimensionality analysis for SAE on {dataset}...")
    compute_dimensionality_matrix('sae', dataset, size_ls, num_models, num_epochs, base_path)
    
    print(f"Running dimensionality analysis for DAE on {dataset}...")
    compute_dimensionality_matrix('dae', dataset, size_ls, num_models, num_epochs, base_path)

    plot_dimensionality_comparison(dataset)
    
    print(f"Dimensionality analysis for {dataset} complete.")


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

cifar_size_ls = [6,   6,   6,   6,   6,   6,    # 6
                10,  10,  10,  10,  10,  10,    # 6
                16,  16,  16,  16,  16,  16,    # 6
                28,  28,  28,  28,  28,  28,    # 6
                48,  48,  48,  48,  48,  48,  48,  48, 48, # 9
                90,  90,  90,  90,  90,  90,  90,  90,  90,  90, #10
                128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                128, 128, 128, 128, 128, 128, 128 # 17
                ]

base_path = '/home/david/'
run_dimensionality_analysis('mnist', mnist_size_ls, num_models=5, num_epochs=30, base_path=base_path)
# run_dimensionality_analysis('cifar', cifar_size_ls, num_models=2, num_epochs=60, base_path=base_path)