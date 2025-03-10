import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


# TODO violin plot? (18)
# TODO plot zero activation in bottleneck barchart (18)
# TODO plot mean activation for bottleneck (18)

layers_to_measure = [ 'encoder_0', 'encoder_2',  'encoder_4',  'encoder_6', 'encoder_8', 'encoder_11']


def get_avg_activations(activations: dict) -> list:
    layer_means = []
    for layer in layers_to_measure:
        mean_activation = torch.mean(activations[layer]).item()
        layer_means.append(mean_activation)
    return layer_means


def get_zero_activations(activations: dict) -> list:
    zero_percentages = []
    for layer in layers_to_measure:
        num_zeros = torch.sum(activations[layer] == 0).item()
        total_elements = activations[layer].numel()
        zero_percent = (num_zeros / total_elements) * 100
        zero_percentages.append(zero_percent)
    
    return zero_percentages


def get_model_activations(model: ConvAutoencoder, image: torch.Tensor) -> list:
    """Get raw activations for each neuron in each layer."""
    with torch.no_grad():
        _, _, activations = model.forward(
            image, 
            return_activations=True
        )

    layer_activations = []
    for layer in layers_to_measure:
        # print('debug: layer', layer)
        # print('debug: activations keys', activations.keys())
        act = activations[layer]
        layer_activations.append(act.detach().cpu())
    
    return layer_activations


def evaluate_model_activations(model_type: str, tensor_test_images: torch.Tensor, ae: ConvAutoencoder) -> list:
    """
    Compute average activations across all test images for each neuron.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        tensor_test_images: Test images to evaluate
        ae: Autoencoder model
        
    Returns:
        list: List of numpy arrays, each containing average activations for neurons in a layer
    """
    # Initialize lists to store activations for each layer
    all_layer_acts = [[] for _ in range(len(layers_to_measure))]
    
    # Process all images
    for img in tensor_test_images:
        img = img.unsqueeze(0)
        layer_acts = get_model_activations(ae, img)
        
        # Store activations for each layer
        for i, acts in enumerate(layer_acts):
            all_layer_acts[i].append(acts)
    
    # Calculate average activations for each layer
    avg_layer_acts = []
    for i in range(len(all_layer_acts)):
        stacked_acts = torch.stack(all_layer_acts[i])
        avg_acts = torch.squeeze(torch.mean(stacked_acts, dim=0)).numpy()
        
        # Pad the activations for the DAE's bottleneck layer
        if model_type == 'dae' and i == 2 and len(avg_acts) < 128:
            padding_size = 128 - len(avg_acts)
            padding = np.full(padding_size, np.nan)
            avg_acts = np.concatenate([avg_acts, padding])
        
        avg_layer_acts.append(avg_acts)
    
    return avg_layer_acts


def compute_activation_for_single_model(model_idx: int, model_type: str, test_images: list, 
                                        num_epochs: int = 60) -> np.ndarray:
    """
    Compute average activations for a single model across all epochs.

    Args:
        model_idx: Index of the model
        model_type: Type of model ('sae' or 'dae')
        test_images: List of test images
        num_epochs: Number of epochs to process

    Returns:
        np.ndarray: Array of average activations with shape (num_epochs, num_layers, max_neurons)
    """
    neurons_per_layer = [ 128]
    
    # Initialize results with shape (num_epochs, num_layers, max_neurons)
    results = np.full((num_epochs, len(layers_to_measure), max(neurons_per_layer)), np.nan)
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        # ae = load_model(f'/home/david/mnist_model/{model_type}/{model_idx}', model_type, epoch)
        run_id = '2025-03-05_13:59:06'
        model_path = f"/home/kong/cifar_models/cnn/{run_id}/{model_type}/{model_idx}"
        ae = load_conv_model(model_path, model_type=model_type, epoch=epoch)
        layer_avgs = evaluate_model_activations(model_type, test_images, ae)
        
        # Store results
        # print('debug: layer_avgs', layer_avgs)
        for layer_idx, layer_avg in enumerate(layer_avgs):
            if len(layer_avg.shape) == 2 and layer_avg.shape[0] == 1:
                layer_avg = layer_avg[0]
            num_neurons = len(layer_avg)
            # print('debug: layer_avg', layer_avg)
            # print('debug: num_neurons', num_neurons)
            results[epoch, layer_idx, :num_neurons] = layer_avg
    
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
        None: Saves activation matrices for all models
    """
    result_file = f'Results/{model_type}_hidden_layer_neuron_activations.npy'
    
    # Check if results already exist
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    # Load test images
    test_images, _ = load_cifar_tensor()
    
    neurons_per_layer = [128]
    
    # Initialize all_results with shape: (num_models, num_epochs, num_layers, max_neurons)
    all_results = np.full((num_models, num_epochs, len(layers_to_measure), max(neurons_per_layer)), np.nan)
    
    # Process all models
    for model_idx in tqdm(range(num_models), desc="Processing models"):
        model_results = compute_activation_for_single_model(
            model_idx, model_type, test_images, num_epochs
        )        
        all_results[model_idx] = model_results
    
    np.save(result_file, all_results)
    
    return None


def plot_neuron_activations(model_type: str, epoch: int = 59) -> None:
    """
    Plot activations given a specific RF and epoch (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        epoch: Epoch to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")

    # Find average and standard deviation of activations for each layer
    avg_layer_acts = np.array([np.nanmean(neuron_activations[:, epoch, i, :]) for i in range(1)])
    std_layer_acts = np.array([np.nanstd(neuron_activations[:, epoch, i, :]) for i in range(1)])

    plt.rc('font', size=16)

    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(avg_layer_acts, label='AE', color='blue', linewidth=4)
    plt.fill_between(range(len(avg_layer_acts)), 
                    avg_layer_acts - std_layer_acts, 
                    avg_layer_acts + std_layer_acts,
                    color='blue', alpha=0.1)

    if model_type == 'sae':
        model_type = 'AE'
    if model_type == 'dae':
        model_type = 'Dev-AE'
    plt.title(f'Neuron Activation Across Hidden Layers ({model_type})')
    plt.ylabel('Mean Activation Value')
    plt.xticks(range(1), ['Bottleneck [128]'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_hidden_layer_neuron_activations_epoch_{epoch}.png", dpi=300)
    plt.close()

    return None


def plot_zero_activations(model_type: str, epoch: int = 59, size_ls: list = None) -> None:
    """
    Plot zero activations (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        epoch: Epoch to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    
    zero_percentages = np.zeros(1)
    
    # Calculate zero percentages for each layer with given epoch
    for layer_idx in range(1):
        all_activations = neuron_activations[:, epoch, layer_idx, :]
        all_activations = np.concatenate(all_activations, axis=0)
        num_zeros = np.sum(all_activations == 0)
        total_elements = np.sum(~np.isnan(all_activations))
        if model_type == 'dae' and layer_idx == 2:
            total_elements = size_ls[epoch]
        zero_percentages[layer_idx] = (num_zeros / total_elements) * 100

    plt.rc('font', size=16)

    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(zero_percentages, label='AE', color='blue', linewidth=4)

    if model_type == 'sae':
        model_type = 'AE'
    if model_type == 'dae':
        model_type = 'Dev-AE'
    plt.title('Percentage of 0s Across Hidden Layers')
    plt.ylabel('Percentage of 0s')
    plt.ylim(0, 100)
    plt.xticks(range(1), ['Bottleneck [128]'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_hidden_layer_zero_activations_epoch_{epoch}.png", dpi=300)
    plt.close()

    return None


def activations_heatmap(model_type: str) -> None:
    """
    Plot heatmap of neuron activations over all epochs.
    """
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    print('debug: neuron_activations', neuron_activations.shape)
    epoch_activations_mean = np.mean(neuron_activations[:, :, 0, :127], axis=0) # CIFAR BOTTLENECK DIFFERENT
    epoch_activations_mean = epoch_activations_mean.T
    print('debug: epoch_activations_mean', epoch_activations_mean.shape)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    heatmap = sns.heatmap(
        epoch_activations_mean[:, :],
        cmap="inferno",
        vmin=0,
        vmax=10,
        cbar_kws={"label": "Angle between PCs"},
        linewidths=0.5,
    )

    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Strength of Activation", fontsize=24)
    cbar.minorticks_off()

    ax.set_xticks([0.5, 24.5, 49.5])
    ax.set_xticklabels(["1-2", "25-26", "49-50"], fontsize=24, rotation=0)

    ax.set_yticks([0.5, 15.5, 31.5])
    ax.set_yticklabels(["1", "16", "32"], fontsize=24, rotation=90)

    ax.set_title(f"Neuron Activation over Epochs ({model_type})", fontsize=28, pad=25)
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Neuron Index", fontsize=24)

    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_neuron_activations_over_time.png", dpi=300)
    plt.close()


def plot_activation_distribution(model_type: str, epoch: int) -> None:
    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    # print('debug: neuron_activations', neuron_activations.shape)

    data = neuron_activations[:, epoch, 0, :128] # CIFAR BOTTLENECK DIFFERENT
    print(data.shape)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for neuron in range(data.shape[1]):
        y_vals = data[:, neuron].flatten() # Flatten data per neuron for all arrays
        x_vals = np.random.normal(loc=neuron, scale=0.1, size=y_vals.shape)
        ax.scatter(x_vals, y_vals, s=1, alpha=0.05)

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Activation Value')
    ax.set_title('DAE Activation Distribution')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_bottleneck_activations.png", dpi=300)
    plt.close()


def compute_hidden_layer_activation(model_type: str, num_models: int = 40, 
                                    num_epochs: int = 60, epoch: int = 59, 
                                    size_ls: list = None) -> None:
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
    plot_neuron_activations(model_type, epoch=epoch)
    plot_zero_activations(model_type, epoch=epoch, size_ls=size_ls)
    activations_heatmap(model_type)
    plot_activation_distribution(model_type, epoch=epoch)
    return None