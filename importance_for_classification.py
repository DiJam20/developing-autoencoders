import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch

from autoencoder import *
from model_utils import *
from solver import *


def encode_dataset(model, dataset_images, dataset="mnist"):
    """
    Encode an entire dataset using the provided model.
    
    Args:
        model: Trained autoencoder model
        dataset_images: Images from the dataset
        dataset: Dataset name ('mnist' or 'cifar')
        
    Returns:
        encodings: Encoded representations
        labels: Corresponding labels
    """
    # Convert to torch tensor if needed
    if isinstance(dataset_images, np.ndarray):
        dataset_images = torch.tensor(dataset_images, dtype=torch.float32)
    
    with torch.no_grad():
        if dataset.lower() == "mnist":
            encoded = model.encode(dataset_images.view(dataset_images.size(0), -1))
        else:
            encoded = model.encode(dataset_images)
    
    return encoded.cpu().numpy()


def get_neuron_importance(classifier):
    """
    Extract and process coefficients from the logistic regression model to determine neuron importance.
    
    Args:
        classifier: Trained LogisticRegression model
        
    Returns:
        importance: Average absolute importance per neuron
    """
    # Extract coefficients
    coeffs = classifier.coef_
    
    # Take the absolute value and average across classes
    importance = np.mean(np.abs(coeffs), axis=0)
    
    return importance


def get_neuron_group_importance(importance, neuron_groups):
    """
    Calculate average importance for each neuron group.
    
    Args:
        importance: Array of neuron importance values
        neuron_groups: List of indices defining the end of each group
        
    Returns:
        List of average importance values for each group
    """
    group_importance = []
    start_indices = [0] + [neuron_groups[i-1] for i in range(1, len(neuron_groups))]
    
    for start_idx, end_idx in zip(start_indices, neuron_groups):
        group_avg = np.mean(importance[start_idx:end_idx])
        group_importance.append(group_avg)
    
    return group_importance


def analyze_model_features(model_index, dataset, base_path, images, labels):
    """
    Analyze neuron importance for a specific model.
    
    Args:
        model_index: Index of the model to analyze
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    
    if dataset.lower() == "mnist":
        model_path = f'{base_path}mnist_models/'
        sae = load_model(f"{model_path}sae/{model_index}", 'sae', 59)
        dae = load_model(f"{model_path}dae/{model_index}", 'dae', 59)
    else:
        model_path = f"{base_path}cifar_models/"
        sae = load_conv_model(f"{model_path}sae/{model_index}", 'sae', 59)
        dae = load_conv_model(f"{model_path}dae/{model_index}", 'dae', 59, size_ls=[128]*60)
        
    sae_train_encodings = encode_dataset(sae, images, dataset)
    dae_train_encodings = encode_dataset(dae, images, dataset)
    
    sae_classifier = LogisticRegression(max_iter=3000)
    dae_classifier = LogisticRegression(max_iter=3000)
    
    sae_classifier.fit(sae_train_encodings, labels)
    dae_classifier.fit(dae_train_encodings, labels)
    
    # Extract neuron importance
    sae_importance = get_neuron_importance(sae_classifier)
    dae_importance = get_neuron_importance(dae_classifier)
    
    return sae_importance, dae_importance


def plot_grouped_importance(sae_importance, dae_importance, neuron_groups, dataset="mnist"):
    """
    Plot neuron importance grouped by neuron groups, side by side for SAE and DAE.
    
    Args:
        sae_importance: SAE neuron importance values
        dae_importance: DAE neuron importance values
        neuron_groups: List of indices defining the end of each group
        dataset: Dataset name ('mnist' or 'cifar')
    """
    sae_group_importance = get_neuron_group_importance(sae_importance, neuron_groups)
    dae_group_importance = get_neuron_group_importance(dae_importance, neuron_groups)
    
    # Create labels for each neuron group
    start_indices = [1] + [neuron_groups[i-1] + 1 for i in range(1, len(neuron_groups))]
    x_labels = [f"{start}-{end}" for start, end in zip(start_indices, neuron_groups)]
    
    plt.rc('font', size=28)
    x_indices = np.arange(len(neuron_groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    # Plot SAE
    sae_bars = ax1.bar(x_indices, sae_group_importance, color='#1a7adb')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Neuron Classification Influence')
    ax1.set_title('AE')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    dae_bars = ax2.bar(x_indices, dae_group_importance, color='#e82817')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    ax2.set_title('DevAE')
    
    max_val = max(max(sae_group_importance), max(dae_group_importance))
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    fig.suptitle('Neuron Group Importance')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(f"Results/figures/png/{dataset}_grouped_neuron_importance", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_grouped_neuron_importance", bbox_inches='tight')
    plt.close()


def compute_neuron_importance(num_models, dataset="mnist", base_path="/home/david/", neuron_groups=None):
    """
    Compute average neuron importance across multiple model iterations and plot grouped importance.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    # Results file for average
    result_file = f"Results/{dataset}_neuron_importance.npy"
    
    # Check if average results already exist
    if os.path.exists(result_file):
        print(f"Loading existing average results from {result_file}")
        avg_results = np.load(result_file, allow_pickle=True).item()
    else:
        all_sae_importance = []
        all_dae_importance = []

        images, labels = load_cifar_list() if dataset.lower() == "cifar" else load_mnist_list()
        
        # Find importance for each model
        for i in tqdm(range(num_models), desc=f"Processing models"):
            sae_importance, dae_importance = analyze_model_features(
                i, dataset, base_path, images, labels
            )
                
            all_sae_importance.append(sae_importance)
            all_dae_importance.append(dae_importance)
        
        # Compute averages
        avg_sae_importance = np.mean(all_sae_importance, axis=0)
        avg_dae_importance = np.mean(all_dae_importance, axis=0)
        
        # Create average results dictionary
        avg_results = {
            'sae_importance': avg_sae_importance,
            'dae_importance': avg_dae_importance,
            'neuron_groups': neuron_groups,
            'num_models': num_models
        }
        
        np.save(result_file, avg_results)

    return avg_results


def run_classification_importance_analysis(num_models, base_path="/home/david/", dataset="mnist", size_ls=None):
    """
    Run analyses.
    
    Args:
        num_models: Number of models to evaluate
        base_path: Base path to the model directory
        dataset: Dataset name ('mnist or 'cifar')
        size_ls: List of bottleneck sizes
    """

    print(f"Running analysis for {dataset}...")
    neuron_groups = sorted(set(size_ls))
    compute_neuron_importance(num_models, dataset, base_path, neuron_groups)
    avg_results = np.load(f"Results/{dataset}_neuron_importance.npy", allow_pickle=True).item()
    plot_grouped_importance(
        avg_results['sae_importance'], 
        avg_results['dae_importance'], 
        avg_results['neuron_groups'], 
        dataset
    )
