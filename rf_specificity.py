import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


def compute_activation_for_single_model(model_idx: int, model_type: str, rf_matrix: np.ndarray,
                                      size_ls: list = None, num_epochs: int = 60) -> list:
    """
    Compute activations for a single model across all epochs.
    
    Args:
        model_idx: Index of the model
        model_type: Type of model ('sae' or 'dae')
        rf_matrix: Pre-loaded receptive fields for this model
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
        
    Returns:
        list: Activation matrix for the model
    """
    activation_matrix = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        ae = load_model(f'/home/david/mnist_model/{model_type}/{model_idx}', model_type, epoch)
        
        num_neurons = size_ls[epoch] if model_type == "dae" else 32
        max_size = 32
        
        # Get RFs for current epoch
        rf_ls = rf_matrix[epoch][:num_neurons]
        
        # Process each RF
        epoch_activations = []
        with torch.no_grad():
            for rf in rf_ls:
                input = torch.tensor(rf, dtype=torch.float32).reshape(1, -1)
                encoded, decoded = ae(input)
                absolute_encoded = torch.abs(encoded)
                # Store as a flat array - critical for consistent shapes
                epoch_activations.append(absolute_encoded.cpu().numpy().flatten())
        
        # Pad to maximum 32 RFs which give maximum 32 activations
        padded_activations = np.zeros((max_size, max_size))

        for i, act in enumerate(epoch_activations):
            if i < max_size:
                # Ensure each activation is padded to max_size length
                act_padded = np.zeros(max_size)
                act_padded[:len(act)] = act[:max_size]  # Truncate or pad as needed
                padded_activations[i] = act_padded
        
        activation_matrix.append(padded_activations)
    
    return activation_matrix


def compute_neuron_activations(model_type: str, num_models: int = 40, 
                               size_ls: list = None, num_epochs: int = 60) -> None:
    """
    Compute activations for all models across all epochs.
    
    Args:
        model_idx: Index of the model
        model_type: Type of model ('sae' or 'dae')
        num_models: Number of models to process
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
    
    Returns:
        None: Saves activation matrices for all models
    """
    result_file = f'Results/{model_type}_rf_neuron_activations.npy'
    
    # Check if results already exist
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file)
    
    # Load all RFs at once
    print(f"Loading all receptive fields...")
    rf_matrices = np.load(f"Results/{model_type}_rfs.npy", allow_pickle=True)
    
    activation_matrices = []
    
    for model_idx in tqdm(range(num_models), desc=f"Processing models"):
        activation_matrix = compute_activation_for_single_model(
            model_idx,
            model_type,
            rf_matrices[model_idx],
            size_ls=size_ls,
            num_epochs=num_epochs
        )
        activation_matrices.append(activation_matrix)
    
    np.save(result_file, np.array(activation_matrices))
    return None


def plot_neuron_activations(model_type: str, epoch: int = 59, neuron_idx: int = 0) -> None:
    """
    Plot activations given a specific RF and epoch (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        epoch: Epoch to plot
        neuron_idx: Index of the neuron to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{model_type}_rf_neuron_activations.npy")
    epoch_activations = neuron_activations[:, epoch, neuron_idx, :]

    mean_activations = np.mean(epoch_activations, axis=0)
    std_activations = np.std(epoch_activations, axis=0)
    plt.figure()
    plt.plot(mean_activations, color='blue')
    plt.fill_between(np.arange(len(mean_activations)),
                     mean_activations - std_activations,
                     mean_activations + std_activations,
                     color='royalblue', alpha=0.2)
    plt.title(f"Activations given Receptive Field {neuron_idx} (Epoch: {epoch})")
    plt.savefig(f"Results/{model_type}_rf_neuron_activations_neurons_{neuron_idx}_epoch_{epoch}.png")
    plt.close()

    return None


def compute_rf_specificity(model_type: str, num_models: int = 40, size_ls: list = None, num_epochs: int = 60) -> None:
    """
    Compute RF specificity for all models.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        num_models: Number of models to process
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
    Results:
        None: Saves results to file
    """
    compute_neuron_activations(model_type, num_models=num_models, size_ls=size_ls, num_epochs=num_epochs)
    plot_neuron_activations(model_type, epoch=59, neuron_idx=0)