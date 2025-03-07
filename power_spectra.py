import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


def z_score(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image using the z-score normalization.
    Normalized image: (pixel - mean) / std

    Args:
        image: 2D numpy array representing the image
    
    Returns:
        normalized_image: 2D numpy array representing the normalized
        image
    """
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image


def radial_profile(data: np.ndarray, center: np.ndarray = None) -> np.ndarray:
    """
    Compute the radial profile of a 2D array data.

    Args:
        data: 2D numpy array representing the image
        center: 1D numpy array representing the center of the image
    
    Returns:
        radialprofile: 1D numpy array representing the radial profile
        of the image
    """
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr

    return radialprofile


def power_spectrum_radial_average(image: np.ndarray) -> np.ndarray:
    """
    Calculate the radial average of the power spectrum for a 2D grey-scale image.
    
    Args:
        image: 2D numpy array representing the image
    Returns:
        radial_avg: radial average of the power spectrum
    """
    # Take the 2D Fourier transform of the image and shift the zero frequency component to the center
    f_transform = np.fft.fftshift(np.fft.fft2(image))

    # Compute the power spectrum (magnitude squared of the Fourier coefficients)
    power_spectrum = np.abs(f_transform) ** 2

    # Compute the radial profile of the power spectrum
    radial_avg = radial_profile(power_spectrum)

    return radial_avg


def rgb_to_grayscale(images):
    """
    Convert RGB images to grayscale.
    
    Parameters:
    images : numpy array of shape (batch_size, 3, height, width)
        Batch of RGB images.
    
    Returns:
    numpy array of shape (batch_size, 1, height, width)
        Batch of grayscale images.
    """
    # Standard luminance formula: 0.299 * R + 0.587 * G + 0.114 * B
    grayscale = 0.299 * images[:, :, :, 0:1, :] + \
                0.587 * images[:, :, :, 1:2, :] + \
                0.114 * images[:, :, :, 2:3, :]
    
    return np.squeeze(grayscale, axis=3)


def load_rfs(save_path_sae: str, save_path_dae:str, num_models: int, epoch: int) -> tuple:
    """
    Load the receptive fields of the models and compute the power spectrum of each RF.
    
    Args:
        num_models: number of models
        epoch: epoch number
        
    Returns:
        sae_power_spectra: list of power spectra of SAE RFs
        dae_power_spectra: list of power spectra of DAE RFs
    """
    sae_rfs = np.load(save_path_sae)
    dae_rfs = np.load(save_path_dae)

    sae_power_spectra = []
    dae_power_spectra = []

    print(dae_rfs.shape)

    MIN_WIDTH = 28
    MIN_HEIGHT = 28

    if len(sae_rfs.shape) == 5:
        sae_rfs = rgb_to_grayscale(sae_rfs)
        dae_rfs = rgb_to_grayscale(dae_rfs)
        MIN_WIDTH = 32
        MIN_HEIGHT = 32

    print(sae_rfs.shape)

    for i in tqdm(range(num_models)):
        sae_power_spectrum = []
        for rf in sae_rfs[i, epoch, :]:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(MIN_WIDTH, MIN_HEIGHT)))
            sae_power_spectrum.append(radial_avg)
        sae_power_spectra.append(sae_power_spectrum)

        dae_power_spectrum = []
        for rf in dae_rfs[i, epoch, :]:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(MIN_WIDTH, MIN_HEIGHT)))
            dae_power_spectrum.append(radial_avg)
        dae_power_spectra.append(dae_power_spectrum)

    return sae_power_spectra, dae_power_spectra


def group_power_spectra(power_spectra, neuron_groups):
    """
    Group power spectra by neuron groups and calculate the average for each group.
    
    Args:
        power_spectra: Power spectra for each neuron, shape (n_neurons, n_frequencies)
        neuron_groups: List of integers representing the end index of each group
                      e.g., [6, 10, 16, 28, 90, 128] means groups are 1-6, 7-10, etc.
    
    Returns:
        grouped_spectra: List of averaged power spectra for each group
        group_labels: Labels for each group (e.g., "1-6", "7-10", etc.)
    """
    grouped_spectra = []
    group_labels = []
    
    start_idx = 0
    for end_idx in neuron_groups:
        # Ensure we don't exceed the number of available neurons
        end_idx = min(end_idx, power_spectra.shape[0])
        
        # Calculate the average power spectrum for this group
        group_avg = np.mean(power_spectra[start_idx:end_idx], axis=0)
        grouped_spectra.append(group_avg)
        
        # Create a label for this group
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
        
        # Update the start index for the next group
        start_idx = end_idx
    
    return np.array(grouped_spectra), group_labels


def plot_power_spectra_subplot(ax, frequency_data, title, group_labels=None, ylim_top=30000):
    """
    Plot power spectra for groups of neurons.
    
    Args:
        ax: Matplotlib axis
        frequency_data: Power spectra data, shape (n_groups, n_frequencies)
        title: Plot title
        group_labels: Labels for each group
        ylim_top: Upper limit for y-axis
    """
    colors = plt.cm.cool(np.linspace(0, 1, frequency_data.shape[0]))
    discrete_cmap = mcolors.ListedColormap(colors)
    
    for idx, freq in enumerate(frequency_data):
        label = f'Neurons {group_labels[idx]}' if group_labels else f'RF{idx + 1}'
        ax.plot(freq, color=colors[idx], label=label)
    
    ax.set(xlabel='Frequency', 
           ylabel='Power',
           xlim=(0, 10),
           ylim=(0, ylim_top))
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title)
    
    if frequency_data.shape[0] <= 10:
        ax.legend(fontsize=10, loc='upper right')
    
    # Only add colorbar for non-grouped version
    if not group_labels:
        bounds = np.arange(-0.5, frequency_data.shape[0] + 0.5, 1)
        norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
        sm.set_array([])

        num_ticks = 3
        tick_positions = np.linspace(0, frequency_data.shape[0]-1, num_ticks, dtype=int)
        tick_labels = [1, frequency_data.shape[0] // 2, frequency_data.shape[0]]

        cbar = plt.colorbar(sm, ax=ax, ticks=tick_positions)
        cbar.set_label('Neuron Index', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_ticklabels(tick_labels)
        cbar.minorticks_off()


def save_power_spectra(sae_power_spectra, dae_power_spectra, neuron_groups=None):
    """
    Plot the power spectra of the receptive fields of the models.

    Args:
        sae_power_spectra: Power spectra for SAE neurons
        dae_power_spectra: Power spectra for DAE neurons
        neuron_groups: List of integers representing the end index of each group
                      e.g., [6, 10, 16, 28, 90, 128] means groups are 1-6, 7-10, etc.
    
    Returns:
        None: saves the plot as a .png file
    """
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    sae_mean = np.mean(sae_power_spectra, axis=0)  # Average across models
    dae_mean = np.mean(dae_power_spectra, axis=0)  # Average across models
    
    if neuron_groups:
        # Group the neurons and plot the average for each group
        sae_grouped, group_labels = group_power_spectra(sae_mean, neuron_groups)
        dae_grouped, _ = group_power_spectra(dae_mean, neuron_groups)
        
        plot_power_spectra_subplot(axs[0], sae_grouped, 'SAE RF Power Spectrum', group_labels)
        plot_power_spectra_subplot(axs[1], dae_grouped, 'DAE RF Power Spectrum', group_labels)
    else:
        # Plot one line per neuron
        plot_power_spectra_subplot(axs[0], sae_mean, 'SAE RF Power Spectrum')
        plot_power_spectra_subplot(axs[1], dae_mean, 'DAE RF Power Spectrum')

    plt.tight_layout()
    plt.savefig('Results/combined_power_spectrum.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_power_spectra(save_path_sae, save_path_dae, num_models, epoch, neuron_groups=None):
    """
    Load RFs, compute power spectra, and plot them with optional neuron grouping.
    
    Args:
        save_path_sae: Path to SAE RF data
        save_path_dae: Path to DAE RF data
        num_models: Number of models
        epoch: Epoch number
        neuron_groups: List of integers representing the end index of each group
                      e.g., [6, 10, 16, 28, 90, 128] means groups are 1-6, 7-10, etc.
    """
    sae_power_spectra, dae_power_spectra = load_rfs(save_path_sae, save_path_dae, num_models, epoch)
    save_power_spectra(sae_power_spectra, dae_power_spectra, neuron_groups)