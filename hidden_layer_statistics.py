import os

import numpy as np
import torch
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


layers_to_measure = [
    'encoder_activation_1',
    'encoder_activation_2',
    'encoder_activation_3',
    'decoder_activation_1',
    'decoder_activation_2'
]


def get_model_activations(model: NonLinearAutoencoder, image: torch.Tensor) -> list:
    """
    Get activations for a model given an image.
    
    Args:
        model: Autoencoder model
        image: Image to evaluate
        
    Returns:
        list: List of activations for each layer
    """
    with torch.no_grad():
        _, _, activations = model.forward(
            image, 
            return_activations=True
        )

    layer_activations = []
    for layer in layers_to_measure:
        act = activations[layer]
        layer_activations.append(act.detach().cpu())
    
    return layer_activations
    

def evaluate_model_activations(tensor_test_images: torch.Tensor, ae: NonLinearAutoencoder) -> list:
    """
    Evaluate per-neuron activations for a model across test images.

    Args:
        tensor_test_images: List of test images to be given to the model
        ae: Autoencoder model
        model_type: Type of model ('sae' or 'dae')

    Returns:
        list: List of activation statistics for each layer
    """
    neurons_per_layer = [512, 128, 32, 128, 512]

    all_layer_acts = [[] for _ in range(len(layers_to_measure))]
    # Process all images
    for img in tensor_test_images:
        layer_acts = get_model_activations(ae, img)
        
        # Store activations for each layer
        for i, acts in enumerate(layer_acts):
            all_layer_acts[i].append(acts)
    
    # Compute statistics for each layer
    results = []
    for i, layer_acts in enumerate(all_layer_acts):
        acts_tensor = torch.stack(layer_acts)
        
        # Calculate activation statistics per neuron
        mean_acts = torch.mean(acts_tensor, dim=0).numpy()
        std_acts = torch.std(acts_tensor, dim=0).numpy()
        
        # Calculate zero percentages per neuron
        zero_tensor = (acts_tensor == 0).float()
        mean_zeros = torch.mean(zero_tensor, dim=0).numpy() * 100
        std_zeros = torch.std(zero_tensor, dim=0).numpy() * 100
        
        layer_stats = np.stack([mean_acts, std_acts, mean_zeros, std_zeros])
            
        results.append(layer_stats)
    
    return results


def compute_activation_for_single_model(model_idx: int, model_type: str, test_images: list, 
                                        num_epochs: int = 60) -> list:
    """
    Compute activations for a single model across all epochs.

    Args:
        model_type: Type of model ('sae' or 'dae')
        test_images: List of test images to be given to the model
        num_epochs: Number of epochs to process

    Returns:
        list: Activation matrix for the model
    """
    
    num_layers = len(layers_to_measure)
    neurons_per_layer = [512, 128, 32, 128, 512]
    
    # Initialize results: (num_epochs, num_layers, 4, max_neurons)
    results = np.zeros((num_epochs, num_layers, 4, max(neurons_per_layer)))
    results.fill(np.nan)
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        ae = load_model(f'/home/david/mnist_model/{model_type}/{model_idx}', model_type, epoch)
        layer_results = evaluate_model_activations(test_images, ae)
        
        # Store results for each layer
        for layer_idx, layer_data in enumerate(layer_results):
            num_neurons = layer_data.shape[1]
            results[epoch, layer_idx, :, :num_neurons] = layer_data
    
    return results


def compute_neuron_activations(model_type: str, num_models: int = 40, 
                               num_epochs: int = 60) -> None:
    """
    Compute activations for all models across all epochs.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        num_models: Number of models to process
        num_epochs: Number of epochs to process
    
    Returns:
        None: Saves activation data for all models
    """
    result_file = f'Results/{model_type}_hidden_layer_neuron_activations.npy'
    
    # Check if results already exist
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    test_images, _ = load_mnist_tensor()
    
    neurons_per_layer = [512, 128, 32, 128, 512]
    
    # Initialize all_results: (num_models, num_epochs, num_layers, 4, neurons_per_layer)
    all_results = np.zeros((num_models, num_epochs, len(layers_to_measure), 4, max(neurons_per_layer)))
    all_results.fill(np.nan)
    
    for model_idx in tqdm(range(num_models), desc="Processing models"):
        model_results = compute_activation_for_single_model(
            model_idx, model_type, test_images, num_epochs
        )
        
        all_results[model_idx] = model_results

    
    # Save results
    np.save(result_file, all_results)
    
    return None


def compute_hidden_layer_activation(model_type: str, num_models: int = 40, 
                                    num_epochs: int = 60) -> None:
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
    compute_neuron_activations(model_type, num_models=num_models, num_epochs=num_epochs)
    return None