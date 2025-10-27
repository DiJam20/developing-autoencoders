import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch
# import umap
from scipy import stats

from autoencoder import *
from model_utils import *
from solver import *

# mpl.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': ['Computer Modern'],
#     'font.size': 11
# })


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
            print(dataset_images.shape)
            encoded = model.encode(dataset_images.reshape(dataset_images.size(0), -1))
        else:
            encoded = model.encode(dataset_images)
    
    return encoded.cpu().numpy()


def create_frequency_noisy_images(images, dataset, noise_scale=1.0):
    """
    Create noisy images with different frequency noise bands.
    
    Args:
        images: Clean images to add noise to
        dataset: Dataset name ('mnist' or 'cifar')
        noise_scale: Scale of the noise to add
        
    Returns:
        Three sets of noisy images corresponding to low, medium, and high frequency noise
    """
    if dataset.lower() == "mnist":
        low_freq = (0, 3)
        mid_freq = (4, 7)
        high_freq = (8, 14)
    else:
        low_freq = (0, 3)
        mid_freq = (4, 7)
        high_freq = (8, 16)
    
    num_samples = len(images)
    
    low_freq_images = np.zeros_like(images)
    mid_freq_images = np.zeros_like(images)
    high_freq_images = np.zeros_like(images)
    
    # Create noisy images
    for idx in range(num_samples):
        original_image = images[idx]
        low_freq_images[idx] = add_frequency_noise(original_image, noise_scale, low_freq[0], low_freq[1])[0]
        mid_freq_images[idx] = add_frequency_noise(original_image, noise_scale, mid_freq[0], mid_freq[1])[0]
        high_freq_images[idx] = add_frequency_noise(original_image, noise_scale, high_freq[0], high_freq[1])[0]
    
    return low_freq_images, mid_freq_images, high_freq_images


def evaluate_frequency_classification(sae_model, pca_ae_model, dae_model, dataset="mnist", noise_scale=1.0):
    """
    Evaluate classification accuracy on images with different frequency noise.
    
    Args:
        sae_model: SAE model
        dae_model: DAE model
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
        noise_scale: Scale of the noise to add
        
    Returns:
        Dictionary with classification accuracies for different frequency types
    """
    if dataset.lower() == "mnist":
        images, labels = load_mnist_list()
    else:
        images, labels = load_cifar_list()
    
    # Create noisy images with different frequency types
    low_freq_images, mid_freq_images, high_freq_images = create_frequency_noisy_images(images, dataset, noise_scale)
    
    # Encode training images
    sae_train_encodings = encode_dataset(sae_model, images[:7000], dataset)
    pca_ae_train_encodings = encode_dataset(pca_ae_model, images[:7000], dataset)
    dae_train_encodings = encode_dataset(dae_model, images[:7000], dataset)
    
    # Encode clean test images
    sae_clean_encodings = encode_dataset(sae_model, images[7000:], dataset)
    pca_ae_clean_encodings = encode_dataset(pca_ae_model, images[7000:], dataset)
    dae_clean_encodings = encode_dataset(dae_model, images[7000:], dataset)
    
    # Train classifiers
    sae_classifier = LogisticRegression(max_iter=3000)
    pca_ae_classifier = LogisticRegression(max_iter=3000)
    dae_classifier = LogisticRegression(max_iter=3000)
    
    sae_classifier.fit(sae_train_encodings, labels[:7000])
    pca_ae_classifier.fit(pca_ae_train_encodings, labels[:7000])
    dae_classifier.fit(dae_train_encodings, labels[:7000])
    
    # Clean test images
    sae_clean_pred = sae_classifier.predict(sae_clean_encodings)
    pca_ae_clean_pred = pca_ae_classifier.predict(pca_ae_clean_encodings)
    dae_clean_pred = dae_classifier.predict(dae_clean_encodings)
    
    sae_clean_acc = accuracy_score(labels[7000:], sae_clean_pred)
    pca_ae_clean_acc = accuracy_score(labels[7000:], pca_ae_clean_pred)
    dae_clean_acc = accuracy_score(labels[7000:], dae_clean_pred)
    
    # Low frequency noise images
    sae_low_freq_encodings = encode_dataset(sae_model, low_freq_images, dataset)
    pca_ae_low_freq_encodings = encode_dataset(pca_ae_model, low_freq_images, dataset)
    dae_low_freq_encodings = encode_dataset(dae_model, low_freq_images, dataset)
    
    sae_low_freq_pred = sae_classifier.predict(sae_low_freq_encodings)
    pca_ae_low_freq_pred = pca_ae_classifier.predict(pca_ae_low_freq_encodings)
    dae_low_freq_pred = dae_classifier.predict(dae_low_freq_encodings)
    
    sae_low_freq_acc = accuracy_score(labels, sae_low_freq_pred)
    pca_ae_low_freq_acc = accuracy_score(labels, pca_ae_low_freq_pred)
    dae_low_freq_acc = accuracy_score(labels, dae_low_freq_pred)
    
    # Mid frequency noise images
    sae_mid_freq_encodings = encode_dataset(sae_model, mid_freq_images, dataset)
    pca_ae_mid_freq_encodings = encode_dataset(pca_ae_model, mid_freq_images, dataset)
    dae_mid_freq_encodings = encode_dataset(dae_model, mid_freq_images, dataset)
    
    sae_mid_freq_pred = sae_classifier.predict(sae_mid_freq_encodings)
    pca_ae_mid_freq_pred = pca_ae_classifier.predict(pca_ae_mid_freq_encodings)
    dae_mid_freq_pred = dae_classifier.predict(dae_mid_freq_encodings)
    
    sae_mid_freq_acc = accuracy_score(labels, sae_mid_freq_pred)
    pca_ae_mid_freq_acc = accuracy_score(labels, pca_ae_mid_freq_pred)
    dae_mid_freq_acc = accuracy_score(labels, dae_mid_freq_pred)
    
    # High frequency noise images
    sae_high_freq_encodings = encode_dataset(sae_model, high_freq_images, dataset)
    pca_ae_high_freq_encodings = encode_dataset(pca_ae_model, high_freq_images, dataset)
    dae_high_freq_encodings = encode_dataset(dae_model, high_freq_images, dataset)
    
    sae_high_freq_pred = sae_classifier.predict(sae_high_freq_encodings)
    pca_ae_high_freq_pred = pca_ae_classifier.predict(pca_ae_high_freq_encodings)
    dae_high_freq_pred = dae_classifier.predict(dae_high_freq_encodings)
    
    sae_high_freq_acc = accuracy_score(labels, sae_high_freq_pred)
    pca_ae_high_freq_acc = accuracy_score(labels, pca_ae_high_freq_pred)
    dae_high_freq_acc = accuracy_score(labels, dae_high_freq_pred)
    
    return {
        'sae_clean_acc': sae_clean_acc,
        'pca_ae_clean_acc': pca_ae_clean_acc,
        'dae_clean_acc': dae_clean_acc,
        'sae_low_freq_acc': sae_low_freq_acc,
        'pca_ae_low_freq_acc': pca_ae_low_freq_acc,
        'dae_low_freq_acc': dae_low_freq_acc,
        'sae_mid_freq_acc': sae_mid_freq_acc,
        'pca_ae_mid_freq_acc': pca_ae_mid_freq_acc,
        'dae_mid_freq_acc': dae_mid_freq_acc,
        'sae_high_freq_acc': sae_high_freq_acc,
        'pca_ae_high_freq_acc': pca_ae_high_freq_acc,
        'dae_high_freq_acc': dae_high_freq_acc
    }


def load_and_calculate_pvalues(dataset="mnist"):
    """
    Load results and calculate p-values for each noise type.
    
    Args:
        dataset: Dataset name ('mnist' or 'cifar')
        
    Returns:
        Dictionary with p-values for each noise type
    """
    # Load all individual results (not just averages)
    all_results_file = f"paper_results/{dataset}_all_frequency_classification.npy"
    
    try:
        all_results = np.load(all_results_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find {all_results_file}")
        print("Make sure you've run the main script to generate all results.")
        return None
    
    # Calculate p-values for each noise type using arrays of individual measurements
    p_values = {}

    # --- SAE vs DAE ---
    _, p_values['clean_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_clean_acc'],
        all_results['dae_clean_acc']
    )
    _, p_values['low_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_low_freq_acc'],
        all_results['dae_low_freq_acc']
    )
    _, p_values['mid_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_mid_freq_acc'],
        all_results['dae_mid_freq_acc']
    )
    _, p_values['high_freq_sae_vs_dae'] = stats.ttest_rel(
        all_results['sae_high_freq_acc'],
        all_results['dae_high_freq_acc']
    )

    # --- SAE vs PCA-AE ---
    _, p_values['clean_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_clean_acc'],
        all_results['pca_ae_clean_acc']
    )
    _, p_values['low_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_low_freq_acc'],
        all_results['pca_ae_low_freq_acc']
    )
    _, p_values['mid_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_mid_freq_acc'],
        all_results['pca_ae_mid_freq_acc']
    )
    _, p_values['high_freq_sae_vs_pca'] = stats.ttest_rel(
        all_results['sae_high_freq_acc'],
        all_results['pca_ae_high_freq_acc']
    )

    # --- DAE vs PCA-AE ---
    _, p_values['clean_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_clean_acc'],
        all_results['pca_ae_clean_acc']
    )
    _, p_values['low_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_low_freq_acc'],
        all_results['pca_ae_low_freq_acc']
    )
    _, p_values['mid_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_mid_freq_acc'],
        all_results['pca_ae_mid_freq_acc']
    )
    _, p_values['high_freq_dae_vs_pca'] = stats.ttest_rel(
        all_results['dae_high_freq_acc'],
        all_results['pca_ae_high_freq_acc']
    )
        
    return p_values


def add_significance_marker(x1, x2, y_start, p_value, ax):
    """
    Adds a significance marker (bracket with star) between two bars.
    """
    if p_value >= 0.05:
        return  # Don't draw for non-significant results

    # Determine star text
    if p_value < 0.001:
        star_text = '***'
    elif p_value < 0.01:
        star_text = '**'
    else:
        star_text = '*'

    # Bracket line
    ax.plot([x1, x1, x2, x2], [y_start, y_start + 0.01, y_start + 0.01, y_start], lw=1, c='black')
    
    # Star text
    ax.text((x1 + x2) * 0.5, y_start + 0.01, star_text, ha='center', va='bottom', color='black', fontsize=11)


def plot_frequency_classification_results(results, dataset="mnist", std_devs=None, p_values=None):
    """
    Plot a bar chart with clean and frequency noise types for SAE and DAE models.
    
    Args:
        results: Results dictionary with accuracies
        dataset: Dataset name ('mnist' or 'cifar')
        std_devs: Dictionary with standard deviations for error bars
        p_values: Dictionary with p-values for significance testing
    """    
    plt.figure(figsize=(6.266 * 0.5, 2.5))
    ax = plt.gca()
    
    sae_accs = [
        results['sae_clean_acc'],
        results['sae_low_freq_acc'],
        results['sae_mid_freq_acc'],
        results['sae_high_freq_acc']
    ]
    pca_ae_accs = [
        results['pca_ae_clean_acc'],
        results['pca_ae_low_freq_acc'],
        results['pca_ae_mid_freq_acc'],
        results['pca_ae_high_freq_acc']
    ]
    dae_accs = [
        results['dae_clean_acc'],
        results['dae_low_freq_acc'],
        results['dae_mid_freq_acc'],
        results['dae_high_freq_acc']
    ]
    
    sae_errors = [std_devs.get(k, 0) for k in ['sae_clean_acc', 'sae_low_freq_acc', 'sae_mid_freq_acc', 'sae_high_freq_acc']]
    pca_ae_errors = [std_devs.get(k, 0) for k in ['pca_ae_clean_acc', 'pca_ae_low_freq_acc', 'pca_ae_mid_freq_acc', 'pca_ae_high_freq_acc']]
    dae_errors = [std_devs.get(k, 0) for k in ['dae_clean_acc', 'dae_low_freq_acc', 'dae_mid_freq_acc', 'dae_high_freq_acc']]
    
    x = np.arange(4)
    width = 0.25
    
    # Create the bar plots
    sae_bars = ax.bar(x - width, sae_accs, width, label='AE', color='#1a7adb', 
                      yerr=sae_errors, capsize=4)
    pca_ae_bars = ax.bar(x, pca_ae_accs, width, label='PCA-AE', color='#34a853',
                         yerr=pca_ae_errors, capsize=4)
    dae_bars = ax.bar(x + width, dae_accs, width, label='Dev-AE', color='#e82817', 
                      yerr=dae_errors, capsize=4)
    
    plt.xlabel('Frequency Noise Type')
    plt.ylabel('Classification Accuracy')
    plt.xticks(x, ['Clean', 'Low', 'Medium', 'High'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance markers if p-values are provided
    if p_values is not None:
        p_keys_sae_dae = ['clean_sae_vs_dae', 'low_freq_sae_vs_dae', 'mid_freq_sae_vs_dae', 'high_freq_sae_vs_dae']
        p_keys_sae_pca = ['clean_sae_vs_pca', 'low_freq_sae_vs_pca', 'mid_freq_sae_vs_pca', 'high_freq_sae_vs_pca']
        p_keys_dae_pca = ['clean_dae_vs_pca', 'low_freq_dae_vs_pca', 'mid_freq_dae_vs_pca', 'high_freq_dae_vs_pca']

        for i in range(4):
            # Determine the top of the bars for positioning the markers
            all_heights = [sae_accs[i] + sae_errors[i], pca_ae_accs[i] + pca_ae_errors[i], dae_accs[i] + dae_errors[i]]
            y_base = max(all_heights) + 0.02

            # SAE vs DAE
            add_significance_marker(x[i] - width, x[i] + width, y_base + 0.05, p_values.get(p_keys_sae_dae[i], 1), ax)
            # SAE vs PCA-AE
            add_significance_marker(x[i] - width, x[i], y_base, p_values.get(p_keys_sae_pca[i], 1), ax)
            # DAE vs PCA-AE
            add_significance_marker(x[i], x[i] + width, y_base, p_values.get(p_keys_dae_pca[i], 1), ax)

    # Adjust y-axis limit to make space for markers
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    
    plt.savefig(f"Results/figures/png/{dataset}_all_freq_classification.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_all_freq_classification.svg", bbox_inches='tight')
    plt.savefig(f"Results/figures/pdf/{dataset}_all_freq_classification.pdf", bbox_inches='tight')
    plt.close()


# def plot_umap_visualizations(sae_model, dae_model, dataset="mnist"):
#     """
#     Create UMAP visualizations for clean and frequency-noisy encodings.
    
#     Args:
#         sae_model: SAE model
#         dae_model: DAE model
#         dataset: Dataset name ('mnist' or 'cifar')
#     """
#     # Load clean test images
#     if dataset.lower() == "mnist":
#         clean_images, labels = load_mnist_list()
#     else:
#         clean_images, labels = load_cifar_list()
    
#     # Create noisy images with different frequency bands
#     low_freq_images, mid_freq_images, high_freq_images, _ = create_frequency_noisy_images(dataset)
    
#     # Encode clean and noisy images
#     sae_clean_encodings, _ = encode_dataset(sae_model, clean_images, labels, dataset)
#     dae_clean_encodings, _ = encode_dataset(dae_model, clean_images, labels, dataset)
    
#     sae_low_freq_encodings, _ = encode_dataset(sae_model, low_freq_images, labels, dataset)
#     dae_low_freq_encodings, _ = encode_dataset(dae_model, low_freq_images, labels, dataset)
    
#     sae_mid_freq_encodings, _ = encode_dataset(sae_model, mid_freq_images, labels, dataset)
#     dae_mid_freq_encodings, _ = encode_dataset(dae_model, mid_freq_images, labels, dataset)
    
#     sae_high_freq_encodings, _ = encode_dataset(sae_model, high_freq_images, labels, dataset)
#     dae_high_freq_encodings, _ = encode_dataset(dae_model, high_freq_images, labels, dataset)
        
#     # Clean encodings
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     sae_clean_umap = reducer.fit_transform(sae_clean_encodings)
    
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     dae_clean_umap = reducer.fit_transform(dae_clean_encodings)
    
#     # Low frequency noise encodings
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     sae_low_freq_umap = reducer.fit_transform(sae_low_freq_encodings)
    
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     dae_low_freq_umap = reducer.fit_transform(dae_low_freq_encodings)
    
#     # Mid frequency noise encodings
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     sae_mid_freq_umap = reducer.fit_transform(sae_mid_freq_encodings)
    
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     dae_mid_freq_umap = reducer.fit_transform(dae_mid_freq_encodings)
    
#     # High frequency noise encodings
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     sae_high_freq_umap = reducer.fit_transform(sae_high_freq_encodings)
    
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     dae_high_freq_umap = reducer.fit_transform(dae_high_freq_encodings)
    
#     plt.figure(figsize=(20, 10))
    
#     # SAE encodings
#     plt.subplot(2, 4, 1)
#     scatter = plt.scatter(sae_clean_umap[:, 0], sae_clean_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("SAE Clean Encodings")
    
#     plt.subplot(2, 4, 2)
#     scatter = plt.scatter(sae_low_freq_umap[:, 0], sae_low_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("SAE Low Frequency Noise")
    
#     plt.subplot(2, 4, 3)
#     scatter = plt.scatter(sae_mid_freq_umap[:, 0], sae_mid_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("SAE Medium Frequency Noise")
    
#     plt.subplot(2, 4, 4)
#     scatter = plt.scatter(sae_high_freq_umap[:, 0], sae_high_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("SAE High Frequency Noise")
    
#     # DAE encodings
#     plt.subplot(2, 4, 5)
#     scatter = plt.scatter(dae_clean_umap[:, 0], dae_clean_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("DAE Clean Encodings")
    
#     plt.subplot(2, 4, 6)
#     scatter = plt.scatter(dae_low_freq_umap[:, 0], dae_low_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("DAE Low Frequency Noise")
    
#     plt.subplot(2, 4, 7)
#     scatter = plt.scatter(dae_mid_freq_umap[:, 0], dae_mid_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("DAE Medium Frequency Noise")
    
#     plt.subplot(2, 4, 8)
#     scatter = plt.scatter(dae_high_freq_umap[:, 0], dae_high_freq_umap[:, 1], c=labels, cmap='Spectral', s=5)
#     plt.colorbar(scatter)
#     plt.title("DAE High Frequency Noise")
    
#     plt.suptitle(f"{dataset.upper()} Encodings UMAP Visualization", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(f"Results/figures/png/{dataset}_all_encodings_umap.png", dpi=300, bbox_inches='tight')
#     plt.savefig(f"Results/figures/svg/{dataset}_all_encodings_umap.svg", bbox_inches='tight')
#     plt.close()


def run_frequency_classification_analysis(iteration, dataset="mnist", base_path="/home/david/", noise_scale=1.0):
    """
    Run the frequency classification analysis for a single model iteration.
    
    Args:
        iteration: Model iteration/index to use
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
        noise_scale: Scale of the noise to add
        
    Returns:
        Results dictionary with accuracies
    """
    if dataset.lower() == "mnist":
        model_path = f'{base_path}mnist_models/'
        sae = load_model(f"{model_path}sae/{iteration}", 'sae', 59)
        dae = load_model(f"{model_path}dae/{iteration}", 'dae', 59)
    else:
        model_path = f"{base_path}cifar_models/"
        sae = load_conv_model(f"{model_path}sae/{iteration}", 'sae', 59)
        pca_ae = load_conv_model(f"{model_path}pca-ae/{iteration}", 'pca-ae', 59)
        dae = load_conv_model(f"{model_path}dev-ae/{iteration}", 'dev-ae', 59, size_ls=[128]*60)
    
    results = evaluate_frequency_classification(sae, pca_ae, dae, dataset, noise_scale)
    
    return results


def compute_average_frequency_classification(num_models, dataset="mnist", base_path="/home/david/", noise_scale=1.0, create_plots=True):
    """
    Compute average frequency classification over multiple model iterations.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
        noise_scale: Scale of the noise to add
        
    Returns:
        A nested dictionary containing comprehensive results.
    """
    result_file = f"paper_results/{dataset}_frequency_classification_summary.npy"
    
    # Check if a summary file already exists
    if os.path.exists(result_file):
        print(f"Loading existing summary results from {result_file}")
        summary_results = np.load(result_file, allow_pickle=True).item()
    else:
        model_keys = ['sae', 'pca_ae', 'dae']
        condition_keys = ['clean', 'low_freq', 'mid_freq', 'high_freq']
        
        # Nested dictionary to hold all results
        summary_results = {
            model: {
                cond: {'all': []} for cond in condition_keys
            } for model in model_keys
        }

        # Evaluate each model
        for i in tqdm(range(num_models), desc=f"Processing models for {dataset}", leave=True):
            results = run_frequency_classification_analysis(i, dataset, base_path, noise_scale)
            
            for model_key in model_keys:
                for cond_key in condition_keys:
                    result_key = f'{model_key}_{cond_key}_acc'
                    if result_key in results:
                        summary_results[model_key][cond_key]['all'].append(results[result_key])

        for model_key in model_keys:
            for cond_key in condition_keys:
                all_data = summary_results[model_key][cond_key]['all']
                if all_data:
                    summary_results[model_key][cond_key]['avg'] = np.mean(all_data)
                    summary_results[model_key][cond_key]['std'] = np.std(all_data)
        
        os.makedirs("paper_results", exist_ok=True)
        np.save(result_file, summary_results)
    
    if create_plots:
        pass
    
    return summary_results


def visualize_frequency_noise(image_idx, dataset="mnist", noise_scale=1, save_path="Results/figures/"):
    """
    Visualize a clean image and its noisy versions with different frequency bands.
    All images will be displayed in grayscale.
    
    Args:
        image_idx: Index of the image to visualize
        dataset: Dataset name ('mnist' or 'cifar')
        noise_scale: Scale of the noise to add
        save_path: Base path to save the figures
        
    Returns:
        None (displays and saves the images)
    """
    if dataset.lower() == "mnist":
        images, _ = load_mnist_list()
        height, width = 28, 28
        low_freq = (0, 3)
        mid_freq = (4, 7)
        high_freq = (8, 14)
    else:
        images, _ = load_cifar_list()
        height, width = 32, 32
        low_freq = (0, 3)
        mid_freq = (4, 7)
        high_freq = (8, 16)
    
    original_image = images[image_idx]

    if dataset.lower() == "cifar":
        # Convert to grayscale 
        original_image = 0.2989 * original_image[0] + 0.5870 * original_image[1] + 0.1140 * original_image[2]

    # Create noisy versions of this image
    low_freq_image, low_noise_scale = add_frequency_noise(original_image, noise_scale, low_freq[0], low_freq[1])
    mid_freq_image, mid_noise_scale = add_frequency_noise(original_image, noise_scale, mid_freq[0], mid_freq[1])
    high_freq_image, high_noise_scale = add_frequency_noise(original_image, noise_scale, high_freq[0], high_freq[1])

    # Calculate MSE loss    
    low_freq_mse = mean_squared_error(original_image.flatten(), low_freq_image.flatten())
    mid_freq_mse = mean_squared_error(original_image.flatten(), mid_freq_image.flatten())
    high_freq_mse = mean_squared_error(original_image.flatten(), high_freq_image.flatten())

    # plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 4, figsize=(6.268*0.9, 1.5))
    
    def plot_image(ax, img, title, mse=0, noise_scale=0):
        if dataset.lower() == "mnist":
            ax.imshow(img.reshape(height, width), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')

        ax.set_title(title, fontsize=11)
        
        if mse != 0:
            # Add MSE loss label
            ax.text(0.5, -0.2, 
                    f"MSE Loss: {mse:.1f}", 
                    horizontalalignment='center', 
                    transform=ax.transAxes
                    )
        
            # Add noise scale label
            ax.text(0.5, -0.4, 
                    f"Noise Scale: {noise_scale:.1f}",
                    horizontalalignment='center',
                    transform=ax.transAxes
                    )
        
        ax.axis('off')
    
    plot_image(axes[0], original_image, f"Original")
    plot_image(axes[1], low_freq_image, f"Low Frequency", low_freq_mse, low_noise_scale)
    plot_image(axes[2], mid_freq_image, f"Medium Frequency", mid_freq_mse, mid_noise_scale)
    plot_image(axes[3], high_freq_image, f"High Frequency", high_freq_mse, high_noise_scale)
    
    # plt.suptitle(f"Image with Different Frequency Noise Types", fontsize=24, y=1.05)
    plt.tight_layout()
    
    plt.savefig(f"{save_path}png/{dataset}_frequency_noise_sample_gray.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}svg/{dataset}_frequency_noise_sample_gray.svg", bbox_inches='tight')
    plt.savefig(f"{save_path}pdf/{dataset}_frequency_noise_sample_gray.pdf", bbox_inches='tight')
    plt.close()


def run_all_frequency_analyses(num_models_mnist, num_models_cifar, base_path="/home/david/", noise_scale=1):
    """
    Run frequency classification analyses for both MNIST and CIFAR datasets.
    
    Args:
        num_models: Number of models to evaluate
        base_path: Base path to the model directory
        noise_scale: Scale of the noise to add
    """
    print("Running frequency classification analysis for MNIST...")
    compute_average_frequency_classification(num_models_mnist, "mnist", base_path, noise_scale)
    visualize_frequency_noise(1, "mnist", noise_scale)

    print("Running frequency classification analysis for CIFAR...")
    compute_average_frequency_classification(num_models_cifar, "cifar", base_path, noise_scale)
    visualize_frequency_noise(1, "cifar", noise_scale)

# compute_average_frequency_classification(1, 'cifar', base_path='/home/david/', noise_scale=1)

if __name__ == "__main__":
    num_models_mnist = 40
    num_models_cifar = 10
    base_path = "/home/david/"
    noise_scale = 1.0

    # compute_average_frequency_classification(num_models_mnist, "mnist", base_path, noise_scale, create_plots=True)
    # compute_average_frequency_classification(num_models_cifar, "cifar", base_path, noise_scale, create_plots=True)

    visualize_frequency_noise(1, "mnist", noise_scale)
    visualize_frequency_noise(1, "cifar", noise_scale)