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
    # Per neuron statistics
    sae_activations_sum = None
    dae_activations_sum = None
    sae_zeros = None
    dae_zeros = None

    # Per image statistics
    sae_per_image_mean = []
    dae_per_image_mean = []
    sae_per_image_zeros = []
    dae_per_image_zeros = []

    num_images = 0
    if isinstance(sae, NonLinearAutoencoder):
        threshold = 0
    else:
        threshold = 1e-4

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
            
            # Per neuron statistics
            sae_activations_sum += sae_activations
            dae_activations_sum += dae_activations
            sae_zeros += (np.abs(sae_activations) <= threshold)
            dae_zeros += (np.abs(dae_activations) <= threshold)

            # Per image statistics
            # Mean activation for all neurons for this image
            sae_per_image_mean.append(np.mean(np.abs(sae_activations)))
            dae_per_image_mean.append(np.mean(np.abs(dae_activations)))
            # Percentage of zeros for this image
            sae_per_image_zeros.append(np.mean(np.abs(sae_activations) < threshold) * 100)
            dae_per_image_zeros.append(np.mean(np.abs(dae_activations) < threshold) * 100)


    # Per neuron statistics
    sae_mean = sae_activations_sum / num_images
    dae_mean = dae_activations_sum / num_images
    sae_zeros_percent = (sae_zeros / num_images) * 100
    dae_zeros_percent = (dae_zeros / num_images) * 100

    # Per image statistics to numpy arrays
    sae_per_image_mean = np.array(sae_per_image_mean)
    dae_per_image_mean = np.array(dae_per_image_mean)
    sae_per_image_zeros = np.array(sae_per_image_zeros)
    dae_per_image_zeros = np.array(dae_per_image_zeros)
    
    return (sae_mean, dae_mean, 
            sae_zeros_percent, dae_zeros_percent,
            sae_per_image_mean, dae_per_image_mean, 
            sae_per_image_zeros, dae_per_image_zeros)


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
    
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_tensor()

    # Per neuron statistics
    all_sae_means = []
    all_dae_means = []
    all_sae_zeros = []
    all_dae_zeros = []

    # Per image statistics
    all_sae_per_image_means = []
    all_dae_per_image_means = []
    all_sae_per_image_zeros = []
    all_dae_per_image_zeros = []

    for iteration in tqdm(range(num_models), desc=f"Evaluating all models", leave=True):
        # Load models based on dataset
        if dataset.lower() == "mnist":
            sae = load_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_model(model_path+'dae/'+str(iteration), 'dae', 59)
        else:
            sae = load_conv_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_conv_model(model_path+'dae/'+str(iteration), 'dae', 59, size_ls=[128]*60)
        
        (sae_mean, dae_mean, 
         sae_zeros, dae_zeros,
         sae_per_image_mean, dae_per_image_mean, 
         sae_per_image_zeros, dae_per_image_zeros) = evaluate_models(test_images, sae, dae)
        
        all_sae_means.append(sae_mean)
        all_dae_means.append(dae_mean)
        all_sae_zeros.append(sae_zeros)
        all_dae_zeros.append(dae_zeros)

        all_sae_per_image_means.append(sae_per_image_mean)
        all_dae_per_image_means.append(dae_per_image_mean)
        all_sae_per_image_zeros.append(sae_per_image_zeros)
        all_dae_per_image_zeros.append(dae_per_image_zeros)

    # Average per neuron statistics
    sae_means_sum = np.zeros_like(all_sae_means[0])
    dae_means_sum = np.zeros_like(all_dae_means[0])
    sae_zeros_sum = np.zeros_like(all_sae_zeros[0])
    dae_zeros_sum = np.zeros_like(all_dae_zeros[0])
    
    for i in range(num_models):
        sae_means_sum += all_sae_means[i]
        dae_means_sum += all_dae_means[i]
        sae_zeros_sum += all_sae_zeros[i]
        dae_zeros_sum += all_dae_zeros[i]
    
    mean_sae = sae_means_sum / num_models
    mean_dae = dae_means_sum / num_models
    mean_sae_zeros = sae_zeros_sum / num_models
    mean_dae_zeros = dae_zeros_sum / num_models
    
    # Average per image statistics
    sae_per_image_means_avg = np.zeros_like(all_sae_per_image_means[0])
    dae_per_image_means_avg = np.zeros_like(all_dae_per_image_means[0])
    sae_per_image_zeros_avg = np.zeros_like(all_sae_per_image_zeros[0])
    dae_per_image_zeros_avg = np.zeros_like(all_dae_per_image_zeros[0])
    
    for i in range(num_models):
        sae_per_image_means_avg += all_sae_per_image_means[i]
        dae_per_image_means_avg += all_dae_per_image_means[i]
        sae_per_image_zeros_avg += all_sae_per_image_zeros[i]
        dae_per_image_zeros_avg += all_dae_per_image_zeros[i]
    
    sae_per_image_means_avg /= num_models
    dae_per_image_means_avg /= num_models
    sae_per_image_zeros_avg /= num_models
    dae_per_image_zeros_avg /= num_models
    
    np.save(result_file, {
        'mean_sae': mean_sae,
        'mean_dae': mean_dae,
        'mean_sae_zeros': mean_sae_zeros,
        'mean_dae_zeros': mean_dae_zeros,
        
        'sae_per_image_means': sae_per_image_means_avg,
        'dae_per_image_means': dae_per_image_means_avg,
        'sae_per_image_zeros': sae_per_image_zeros_avg,
        'dae_per_image_zeros': dae_per_image_zeros_avg
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
    
    if dataset.lower() == "mnist":
        plt.figure(figsize=(6, 4))
    else:
        plt.figure(figsize=(12, 6))

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
    Plot the percentage of zero activations per neuron for the SAE and DAE models,
    grouped by neuron ranges and displayed in separate subplots.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [4, 10, 16, 24, 32]
    elif dataset.lower() == "cifar":
        neuron_groups = [6, 10, 16, 28, 48, 90, 128]
        
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros'].squeeze()
    mean_dae_zeros = results['mean_dae_zeros'].squeeze()
    
    # Calculate start indices for each neuron group
    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    # Create labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean for each group
    sae_group_means = []
    dae_group_means = []
    sae_group_stds = []
    dae_group_stds = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = mean_sae_zeros[start:end]
        dae_group = mean_dae_zeros[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        dae_group_stds.append(np.std(dae_group))
    
    plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    # Plot SAE
    x_indices = np.arange(len(neuron_groups))
    sae_bars = ax1.bar(x_indices, sae_group_means, color='#1a7adb', yerr=sae_group_stds, capsize=5)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Percentage of Zero Activations')
    ax1.set_xlabel('Neuron Group')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('AE', fontsize=22)
    
    # Plot DAE
    dae_bars = ax2.bar(x_indices, dae_group_means, color='#e82817', yerr=dae_group_stds, capsize=5)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel('Neuron Group')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    ax2.set_title('DevAE', fontsize=22)
    
    # Set the same y-limit for both plots
    max_val = max(
        max(sae_group_means) + max(sae_group_stds),
        max(dae_group_means) + max(dae_group_stds)
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    fig.suptitle(f'Neuron Activation Sparsity by Group ({dataset.upper()})', fontsize=24, y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(f'Results/figures/png/{dataset}_zeros_per_neuron_grouped.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_zeros_per_neuron_grouped.svg', bbox_inches='tight')
    plt.close()


def plot_per_image_activation_distribution(dataset: str):
    """
    Plot the distribution of mean activations per image for SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    
    sae_per_image_means = results['sae_per_image_means']
    dae_per_image_means = results['dae_per_image_means']
    
    plt.rc('font', size=16)
    plt.figure(figsize=(10, 6))
    
    plt.hist(sae_per_image_means, alpha=0.6, label='AE', color='#1a7adb')
    plt.hist(dae_per_image_means, alpha=0.6, label='DevAE', color='#e82817')
    
    plt.xlabel('Mean Activation per Image')
    plt.ylabel('Number of Images')
    plt.title(f'Neuron Activation per Image ({dataset.upper()})', pad=20)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_per_image_activation_dist.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_per_image_activation_dist.svg', bbox_inches='tight')
    plt.close()


def plot_per_image_zeros_distribution(dataset: str):
    """
    Plot the distribution of percentage of zeros per image for SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    
    sae_per_image_zeros = results['sae_per_image_zeros']
    dae_per_image_zeros = results['dae_per_image_zeros']
    
    plt.rc('font', size=16)
    plt.figure(figsize=(10, 6))
    
    plt.hist(sae_per_image_zeros, alpha=0.6, label='AE', color='#1a7adb')
    plt.hist(dae_per_image_zeros, alpha=0.6, label='DevAE', color='#e82817')
    
    plt.xlabel('Percentage of Zero Activations per Image')
    plt.ylabel('Number of Images')
    plt.title(f'Neuron Sparsity per Image ({dataset.upper()})', pad=20)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_per_image_zeros_dist.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_per_image_zeros_dist.svg', bbox_inches='tight')
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
    
    plot_per_image_activation_distribution(dataset)
    plot_per_image_zeros_distribution(dataset)