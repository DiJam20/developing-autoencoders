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


def load_rfs(num_models: int, epoch: int) -> tuple:
    """
    Load the receptive fields of the models and compute the power spectrum of each RF.
    
    Args:
        num_models: number of models
        epoch: epoch number
        
    Returns:
        sae_power_spectra: list of power spectra of SAE RFs
        dae_power_spectra: list of power spectra of DAE RFs
    """
    sae_rf_all = []
    dae_rf_all = []

    sae_power_spectra = []
    dae_power_spectra = []

    for i in tqdm(range(num_models)):
        sae_rfs = np.load('Results/sae_rfs.npy')[i, epoch, :]
        dae_rfs = np.load('Results/dae_rfs.npy')[i, epoch, :]

        sae_power_spectrum = []
        for rf in sae_rfs:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(28, 28)))
            sae_power_spectrum.append(radial_avg)
        sae_power_spectra.append(sae_power_spectrum)

        dae_power_spectrum = []
        for rf in dae_rfs:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(28, 28)))
            dae_power_spectrum.append(radial_avg)
        dae_power_spectra.append(dae_power_spectrum)

    sae_rf_all = np.array(sae_rf_all).squeeze()
    dae_rf_all = np.array(dae_rf_all).squeeze()

    return sae_power_spectra, dae_power_spectra


def plot_power_spectra_subplot(ax, frequency_data: np.ndarray, title: str, ylim_top: int = 16000):
    colors = plt.cm.cool(np.linspace(0, 1, 32))
    discrete_cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, 32.5, 1)
    
    for idx, freq in enumerate(frequency_data):
        ax.plot(freq, color=colors[idx], label=f'RF{idx + 1}')
    
    ax.set(xlabel='Frequency', 
           ylabel='Power',
           xlim=(0, 10),
           ylim=(0, ylim_top))
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(title)
    
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, ticks=[0, 10, 20, 30])
    cbar.set_label('Neuron Index', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.minorticks_off()


def save_power_spectra(sae_power_spectra: list, dae_power_spectra: list) -> None:
    """
    Plot the power spectra of the receptive fields of the models.

    Args:
        num_models: number of models
        epoch: epoch number
    
    Returns:
        None: saves the plot as a .png file
    """

    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    plot_power_spectra_subplot(axs[0], np.mean(sae_power_spectra, axis=0), 'SAE RF Power Spectrum')
    plot_power_spectra_subplot(axs[1], np.mean(dae_power_spectra, axis=0), 'DAE RF Power Spectrum')

    plt.tight_layout()
    plt.savefig('Results/combined_power_spectrum.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_power_spectra(num_models: int, epoch: int) -> None:
    sae_power_spectra, dae_power_spectra = load_rfs(num_models, epoch)
    save_power_spectra(sae_power_spectra, dae_power_spectra)