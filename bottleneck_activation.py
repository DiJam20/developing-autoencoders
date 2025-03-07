import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


def get_bottleneck_activation(model: NonLinearAutoencoder, img: torch.Tensor) -> np.ndarray:
    """
    Get the bottleneck activation of a model for a given image.
    
    Args:
        model: The autoencoder model
        img: The input image
        
    Returns:
        The bottleneck activation
    """
    original_encoding = model.encode(img)
    return original_encoding.detach().numpy()    


def evaluate_models(test_images: np.ndarray, sae: NonLinearAutoencoder, dae: NonLinearAutoencoder) -> tuple:
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
            sae_diff = get_bottleneck_activation(sae, test_image)
            dae_diff = get_bottleneck_activation(dae, test_image)

            if sae_activations_sum is None:
                sae_activations_sum = np.zeros_like(sae_diff)
                dae_activations_sum = np.zeros_like(dae_diff)
                sae_zeros = np.zeros_like(sae_diff)
                dae_zeros = np.zeros_like(dae_diff)
            
            sae_activations_sum += sae_diff
            dae_activations_sum += dae_diff
            
            sae_zeros += (sae_diff == 0)
            dae_zeros += (dae_diff == 0)

    # Calculate mean activations
    sae_mean = sae_activations_sum / num_images
    dae_mean = dae_activations_sum / num_images
    
    # Calculate percentage of zeros
    sae_zeros_percent = (sae_zeros / num_images) * 100
    dae_zeros_percent = (dae_zeros / num_images) * 100
    
    return sae_mean, dae_mean, sae_zeros_percent, dae_zeros_percent


def compute_bottleneck_activation(num_models: int):
    """
    Compute the bottleneck activation for all models and save the results to a file.
    """
    result_file = "Results/bottleneck_activation.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    test_images, _ = load_mnist_tensor()

    all_sae_means = []
    all_dae_means = []
    all_sae_zeros = []
    all_dae_zeros = []

    for iteration in tqdm(range(num_models), desc=f"Evaluating all models", leave=False):
        modelpath = f'/home/david/mnist_model/'
        sae = load_model(modelpath+'sae/'+str(iteration), 'sae', 59)
        dae = load_model(modelpath+'dae/'+str(iteration), 'dae', 59)
        
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


def plot_activation_per_neuron():
    """
    Plot the mean activation per neuron for the SAE and DAE models.
    """
    result_file = "Results/bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    sae_mean = results['mean_sae']
    dae_mean = results['mean_dae']

    plt.rc('font', size=16)
    plt.figure(figsize=(6, 4), dpi=300)

    x = np.arange(32)
    plt.plot(x, sae_mean, label='AE', color='blue', linewidth=2)

    x = np.arange(32)
    plt.plot(x, dae_mean, label='Dev-AE', color='red', linewidth=2)

    plt.xlabel('Neuron Index')
    plt.ylabel('Activation')
    plt.title('Mean Activation per Neuron', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('Results/activation_per_neuron.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_zeros_per_neuron():
    """
    Plot the percentage of zero activations per neuron for the SAE and DAE models.
    """
    result_file = "Results/bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros']
    mean_dae_zeros = results['mean_dae_zeros']

    neurons = np.arange(32)

    plt.rc('font', size=16)
    plt.figure(figsize=(12, 6), dpi=300)

    bar_width = 0.4

    plt.bar(neurons - bar_width/2, mean_sae_zeros, color='blue', width=bar_width, label='AE')
    plt.bar(neurons + bar_width/2, mean_dae_zeros, color='red', width=bar_width, label='Dev-AE')

    plt.xlabel('Neuron Index')
    plt.ylabel('Percentage of Zero Activations')
    plt.title('Zero Activation per Neuron', pad=20)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('Results/zeros_per_neuron.png', bbox_inches='tight', dpi=300)

    plt.show()


def run_bottleneck_activation_analysis(num_models: int):
    compute_bottleneck_activation(num_models)
    plot_activation_per_neuron()
    plot_zeros_per_neuron()