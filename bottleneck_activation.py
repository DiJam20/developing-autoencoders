import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


def get_bottleneck_activation(model, img: torch.Tensor) -> np.ndarray:
    """
    Get the bottleneck activation of a model for a given image.
    
    Args:
        model: The autoencoder model
        img: The input image
        
    Returns:
        The bottleneck activation
    """
    if isinstance(model, ConvAutoencoder):
        original_encoding = model.encode(img.reshape(1, 3, 32, 32))
    else:
        original_encoding = model.encode(img)
    return original_encoding.detach().numpy()    


def evaluate_models(test_images: np.ndarray, sae, dae) -> tuple:
    """
    Evaluate the models by calculating the mean bottleneck activation and the percentage of zeros for each neuron.

    Args:
        test_images: The test images
        sae: The SAE model
        dae: The DAE model

    Returns:
        Tuple containing the mean bottleneck activation for the SAE and DAE models, and the percentage of zeros for each neuron
    """
    sae_activations_sum = None
    dae_activations_sum = None
    sae_zeros = None
    dae_zeros = None
    num_images = 0
    
    for i in range(len(test_images)):
        test_image = test_images[i]
        num_images += 1
        
        with torch.no_grad():
            sae_activations = get_bottleneck_activation(sae, test_image)
            dae_activations = get_bottleneck_activation(dae, test_image)

            if sae_activations_sum is None:
                sae_activations_sum = np.zeros_like(sae_activations)
                dae_activations_sum = np.zeros_like(dae_activations)
                sae_zeros = np.zeros_like(sae_activations)
                dae_zeros = np.zeros_like(dae_activations)
            
            sae_activations_sum += sae_activations
            dae_activations_sum += dae_activations
            
            # sae_zeros += (sae_activations == 0)
            # dae_zeros += (dae_activations == 0)

            threshold = 1e-3
            sae_zeros += (np.abs(sae_activations) < threshold)
            dae_zeros += (np.abs(dae_activations) < threshold)

    # Calculate mean activations
    sae_mean = sae_activations_sum / num_images
    dae_mean = dae_activations_sum / num_images
    
    # Calculate percentage of zeros
    sae_zeros_percent = (sae_zeros / num_images) * 100
    dae_zeros_percent = (dae_zeros / num_images) * 100
    
    return sae_mean, dae_mean, sae_zeros_percent, dae_zeros_percent


def compute_bottleneck_activation(num_models: int, dataset: str, base_path: str):
    """
    Compute the bottleneck activation for all models and save the results to a file.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    if dataset.lower() == "mnist":
        model_path = f'{base_path}mnist_models/'
    elif dataset.lower() == "cifar":
        model_path = f"{base_path}cifar_models/"
    
    result_file = f"Results/{dataset}_bottleneck_activation.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    # Load test images based on dataset
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_tensor()

    all_sae_means = []
    all_dae_means = []
    all_sae_zeros = []
    all_dae_zeros = []

    for iteration in tqdm(range(num_models), desc=f"Evaluating all models", leave=False):
        # Load models based on dataset
        if dataset.lower() == "mnist":
            sae = load_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_model(model_path+'dae/'+str(iteration), 'dae', 59)
        else:
            sae = load_conv_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_conv_model(model_path+'dae/'+str(iteration), 'dae', 59, size_ls=[128]*60)
        
        sae_mean, dae_mean, sae_zeros, dae_zeros = evaluate_models(test_images, sae, dae)
        all_sae_means.append(sae_mean)
        all_dae_means.append(dae_mean)
        all_sae_zeros.append(sae_zeros)
        all_dae_zeros.append(dae_zeros)

    # Calculate mean across all models
    sae_means_sum = all_sae_means[0].copy()
    dae_means_sum = all_dae_means[0].copy()
    sae_zeros_sum = all_sae_zeros[0].copy()
    dae_zeros_sum = all_dae_zeros[0].copy()
    
    for i in range(1, num_models):
        sae_means_sum += all_sae_means[i]
        dae_means_sum += all_dae_means[i]
        sae_zeros_sum += all_sae_zeros[i]
        dae_zeros_sum += all_dae_zeros[i]
    
    mean_sae = sae_means_sum / num_models
    mean_dae = dae_means_sum / num_models
    mean_sae_zeros = sae_zeros_sum / num_models
    mean_dae_zeros = dae_zeros_sum / num_models
    
    np.save(result_file, {
        'mean_sae': mean_sae,
        'mean_dae': mean_dae,
        'mean_sae_zeros': mean_sae_zeros,
        'mean_dae_zeros': mean_dae_zeros
    })


def plot_activation_per_neuron(dataset: str):
    """
    Plot the mean activation per neuron for the SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        MAX_NEURONS = 32
    elif dataset.lower() == "cifar":
        MAX_NEURONS = 128
        
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    sae_mean = abs(results['mean_sae'].squeeze())
    dae_mean = abs(results['mean_dae'].squeeze())

    plt.rc('font', size=16)
    
    # Adjust figure size based on number of neurons
    if dataset.lower() == "mnist":
        plt.figure(figsize=(6, 4), dpi=300)
    else:  # cifar
        plt.figure(figsize=(12, 6), dpi=300)

    x = np.arange(1, MAX_NEURONS + 1)
    plt.plot(x, sae_mean, label='AE', color='#1a7adb', linewidth=2)
    plt.plot(x, dae_mean, label='DevAE', color='#e82817', linewidth=2)

    plt.xlabel('Neuron Index')
    plt.ylabel('Activation')
    plt.title(f'Mean Activation per Neuron ({dataset.upper()})', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(f'Results/figures/png/{dataset}_activation_per_neuron.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_activation_per_neuron.svg', bbox_inches='tight')
    plt.close()


def plot_zeros_per_neuron(dataset: str = "mnist"):
    """
    Plot the percentage of zero activations per neuron for the SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        MAX_NEURONS = 32
        fig_size = (12, 6)
    elif dataset.lower() == "cifar":
        MAX_NEURONS = 128
        fig_size = (18, 6)
        
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros'].squeeze()
    mean_dae_zeros = results['mean_dae_zeros'].squeeze()
    print(mean_sae_zeros.shape)
    print(mean_dae_zeros)

    neurons = np.arange(1, MAX_NEURONS + 1)

    plt.rc('font', size=16)
    plt.figure(figsize=fig_size, dpi=300)

    bar_width = 0.4

    plt.bar(neurons - bar_width/2, mean_sae_zeros, color='#1a7adb', width=bar_width, label='AE')
    plt.bar(neurons + bar_width/2, mean_dae_zeros, color='#e82817', width=bar_width, label='DevAE')

    plt.xlabel('Neuron Index')
    plt.ylabel('Percentage of Zero Activations')
    plt.title(f'Neuron Activation Sparsity Across the Bottleneck Layer ({dataset.upper()})', pad=20)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(f'Results/figures/png/{dataset}_zeros_per_neuron.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_zeros_per_neuron.svg', bbox_inches='tight')
    plt.close()


def run_bottleneck_activation_analysis(num_models: int, dataset: str, base_path: str):
    """
    Run the complete bottleneck activation analysis.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    compute_bottleneck_activation(num_models, dataset, base_path)
    plot_activation_per_neuron(dataset)
    plot_zeros_per_neuron(dataset)