import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from model_utils import cosine_angle_between_pcs, load_model


def initialize_mnist_dataset():
    """
    Load MNIST test data and normalise it.

    Returns:
        np.array: List of MNIST images
        np.array: List of MNIST labels
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=6)

    test_images = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_images.append(data)
        test_labels.append(target)

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    test_images = test_images.squeeze(1)

    return test_images, test_labels


def calculate_angles_for_single_model(model_idx, model_type, size_ls=None, num_epochs=60):
    """
    Calculate PCA angles for a single model across all epochs.
    
    Args:
        model_idx: Index of the model to process
        model_type: Type of model ('sae' or 'dae')
        size_ls: List of component sizes for DAE
        num_epochs: Number of epochs to process
        
    Returns:
        list: Angles between consecutive PCs for this model
    """
    # Load test data
    test_images, _ = initialize_mnist_dataset()
    
    latent_matrices = []
    
    # Calculate latent matrices for each epoch
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        latent_matrix = []
        model_path = f"/home/david/mnist_model/{model_type}/{model_idx}"
        ae = load_model(model_path, model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        for image in test_images:
            image = torch.tensor(image, dtype=torch.float32).reshape(-1)
            
            with torch.no_grad():
                encoded, _ = ae(image)
                latent_matrix.append(encoded.detach().numpy())
        
        latent_matrix = np.stack(latent_matrix)
        
        # Perform PCA on all available bottleneck neurons
        if model_type == "sae":
            pca = PCA(n_components=32)
            pca.fit(latent_matrix)
            pca_components = np.array(pca.components_)
        elif model_type == "dae":
            pca = PCA(n_components=size_ls[epoch])
            pca.fit(latent_matrix)
            pca_components = np.pad(pca.components_, (0, 32 - pca.components_.shape[0]), 'constant')
            
        latent_matrices.append(pca_components)
    
    # Calculate angles between consecutive epochs
    angles_per_model = []
    for i in range(32):
        angles_per_pc = []
        for j in range(num_epochs-1):
            cosine_angle = cosine_angle_between_pcs(latent_matrices[j][i], latent_matrices[j+1][i])
            angles_per_pc.append(cosine_angle)
        angles_per_model.append(angles_per_pc)
    
    return angles_per_model


def compute_angle_matrix(model_type, size_ls=None, num_models=10, num_epochs=60):
    """
    Calculate principal component angles across epochs for multiple models using parallel processing.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        size_ls: List of component sizes for DAE models
        num_models: Number of models to process
        num_epochs: Number of epochs per model
        
    Returns:
        None: Results are saved to a file
    """    
    # Get number of CPU cores (leave one free for system processes)
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Starting processing of {num_models} {model_type} models using {max_workers} workers")

    result_file = f"Results/{model_type}_pc_stability_all_angles.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file)
    
    # Create process pool and map the function across all models
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the function across all model indices with a progress bar
        futures = []
        for idx in range(num_models):
            future = executor.submit(calculate_angles_for_single_model, idx, model_type, size_ls, num_epochs)
            futures.append(future)
            
        # Process results with progress bar
        all_angles = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_models, desc=f"Processing {model_type} models"):
            result = future.result()
            all_angles.append(result)
    
    # Save results
    np.save(result_file, all_angles)
    print(f"Results saved to {result_file}")


def compute_average_angle_matrix(model_type):
    all_angles = np.load(f"Results/{model_type}_pc_stability_all_angles.npy")
    
    if model_type == "sae":
        average_all_angles = np.mean(all_angles, axis=0)
        angle_matrix = np.array(average_all_angles)
        non_computable_cells = np.zeros_like(angle_matrix)
    elif model_type == "dae":
        average_all_angles = np.mean(all_angles, axis=0)
        angle_matrix = np.array(average_all_angles)
        highlighted_non_computable_angles = angle_matrix.copy()

        for i in range(highlighted_non_computable_angles.shape[0]):
            for j in range(highlighted_non_computable_angles.shape[1] - 1):
                if np.isnan(highlighted_non_computable_angles[i, j]) and not np.isnan(highlighted_non_computable_angles[i, j + 1]):
                    highlighted_non_computable_angles[i, j] = 100

        mask = highlighted_non_computable_angles == 100
        non_computable_cells = np.where(mask, 1, np.nan)

    return angle_matrix, non_computable_cells


def create_heatmap(angle_matrix, non_computable_cells, model_type):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    heatmap = sns.heatmap(
        angle_matrix[:, :],
        cmap="plasma",
        vmin=0,
        vmax=90,
        cbar_kws={"label": "Angle between PCs"},
        linewidths=0.5,
        square=True,
    )

    # Grey out non-computable cells
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

    # Customize colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0°", "45°", "90°"], fontsize=24)
    cbar.set_label("PC Angle Difference", fontsize=24)
    cbar.minorticks_off()

    # Customize axes
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
        ax.set_title("Stability of Principal Components (AE)", fontsize=28, pad=25)
    elif model_type == "dae":
        ax.set_title("Stability of Principal Components (Dev-AE)", fontsize=28, pad=25)
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Principal Component Index", fontsize=24)

    plt.savefig(f"Results/{model_type}_stability_of_pcs.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"Results/{model_type}_stability_of_pcs.svg", bbox_inches="tight")
    plt.close(fig)


def analyze_pc_stability(model_type, size_ls=None, num_models=10, num_epochs=60):
    """
    Main function to perform complete PC stability analysis.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        size_ls: List of component sizes for DAE models
        num_models: Number of models to process
        num_epochs: Number of epochs per model
    """
    print(f"Starting PC stability analysis for {model_type} model type")
    
    # Step 1: Compute angle matrix
    compute_angle_matrix(model_type, size_ls, num_models, num_epochs)
    
    # Step 2: Compute average angle matrix
    angle_matrix, non_computable_cells = compute_average_angle_matrix(model_type)
    
    # Step 3: Create heatmap visualization
    create_heatmap(angle_matrix, non_computable_cells, model_type)
    
    print(f"PC stability analysis for {model_type} complete.")
