import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import os, sys
import re
import scipy
from scipy.fftpack import fftshift, fft2
from tqdm import tqdm
import argparse

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image


def z_score(image):
	"""
	Perform Z-score normalization (mean = 0, standard deviation = 1).

	Args:
		image (2D numpy array): Grayscale image.

	Returns:
		2D numpy array: Z-score normalized image.
	"""
	img_mean = np.mean(image)
	img_std = np.std(image)

	# Avoid division by zero if all pixel values are the same
	if img_std == 0:
		return np.zeros_like(image)

	normalized_image = (image - img_mean) / img_std

	return normalized_image

def radial_profile(data, center=None):
    """
    Compute the radial profile of a 2D array `data`.
    """
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr
    return radialprofile

def power_spectrum_radial_average(image):
    """
    Calculate the radial average of the power spectrum for a 2D grey-scale image.
    :param image: 2D numpy array representing the image
    :return: radial average of the power spectrum
    """
    # Take the 2D Fourier transform of the image and shift the zero frequency component to the center
    f_transform = fftshift(fft2(image))
    
    # Compute the power spectrum (magnitude squared of the Fourier coefficients)
    power_spectrum = np.abs(f_transform) ** 2

    # Compute the radial profile of the power spectrum
    radial_avg = radial_profile(power_spectrum)

    return radial_avg

def find_last_epoch(directory):
	latest_epoch = -1
	# List all files in the specified directory
	files = os.listdir(directory)
	for file in files:
		# Check if the file is a .pth file
		if file.endswith('.pth'):
			# Use regex to extract the epoch number
			match = re.search(r'epoch(\d+)', file)
			if match:
				epoch = int(match.group(1))
				# Update if this epoch is the latest
				if epoch > latest_epoch:
					latest_epoch = epoch
						
	return latest_epoch



if __name__ == "__main__":
	## Set font size
	matplotlib.rcParams.update({'font.size': 16})

	run_id = '2025-02-13_19:45:40'
	modelpath = os.getenv("HOME")+'/cifar_models/{}/'.format(run_id)
	plotpath = os.getenv("HOME")+'/cifar_plots/RF/{}/'.format(run_id)
	if not os.path.exists(plotpath):
		os.makedirs(plotpath)

	sae_rf_ls = np.load(modelpath + 'sae_maxact.npy' )
	dae_rf_ls = np.load(modelpath + 'dae_maxact.npy' )
	
	srf_freq_ls = []
	for srf in sae_rf_ls:
		radial_avg = power_spectrum_radial_average(z_score(srf))
		srf_freq_ls.append(radial_avg)

	drf_freq_ls = []
	for drf in dae_rf_ls:
		radial_avg = power_spectrum_radial_average(z_score(drf))
		drf_freq_ls.append(radial_avg)

	np.save(modelpath + 'sae_rf_freq.npy',srf_freq_ls)
	np.save(modelpath + 'dae_rf_freq.npy',drf_freq_ls)

	# plot the radial power spectrum
	colors = plt.cm.cool(np.linspace(0,1,256))
	plot = plt.figure(figsize=(6,5))
	for i,freq in enumerate(srf_freq_ls):
		plt.plot(freq,color=colors[i],label= 'RF'+str(i+1))
	plt.xlabel('Frequency')
	plt.ylabel('Power')
	plt.title('SAE Power Spectrum')
	plt.legend()
	plt.savefig(plotpath + 'sae_psd.pdf',bbox_inches='tight')
	plt.close(plot)

	plot = plt.figure(figsize=(6,5))
	for i,freq in enumerate(drf_freq_ls):
		plt.plot(freq,color=colors[i],label= 'RF'+str(i+1))
	plt.xlabel('Frequency')
	plt.ylabel('Power')
	plt.title('DAE Power Spectrum')
	plt.legend()
	plt.savefig(plotpath + 'dae_psd.pdf',bbox_inches='tight')

	# plot the radial frequency
	plot = plt.figure(figsize=(8,6))
	for i in range(256):
		plt.plot(srf_freq_ls[i],label='SAE')
		plt.plot(drf_freq_ls[i],label='DAE')
	plt.legend()
	plt.xlabel('Frequency')
	plt.ylabel('Power')
	plt.title('Radial Frequency')
	plt.savefig(plotpath + 'freq_comparison.pdf',bbox_inches='tight')
	plt.close(plot)