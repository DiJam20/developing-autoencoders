import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm
import argparse

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import act_max_util as amu

from autoencoder import NonLinearAutoencoder
from solver import *
from receptive_field import get_RF_linear


def load_model(model_path,model_type, n_layers,n_hidden_ls,size_ls, epoch):
	n_input = 3072
	sae_n_hidden_ls = n_hidden_ls

	dae_n_hidden_ls = n_hidden_ls
	print(n_hidden_ls[:-1])
	print(size_ls[epoch])
	print(dae_n_hidden_ls)

	hidden_layers = sae_n_hidden_ls if model_type == 'sae' else dae_n_hidden_ls
	model = NonLinearAutoencoder(n_input, hidden_layers, n_layers)

	weights = torch.load(f"{model_path}/model_weights_epoch{epoch}.pth")
	model.load_state_dict(weights)
	return model

def normalize_image(img,n_type='one'):
	if n_type == 'one':
		ma = img.max()
		mi = img.min()
		res = (img - mi)/(ma-mi)
	if n_type == '255':
		ma = img.max()
		mi = img.min()
		res = 255* (img - mi)/(ma-mi)

	return res


if __name__ == '__main__':
	## load model
	run_id = '2025-02-13_19:45:40'
	modelpath = os.getenv("HOME")+'/cifar_models/{}/'.format(run_id)
	plotpath = os.getenv("HOME")+'/cifar_plots/RF/{}/'.format(run_id)
	if not os.path.exists(plotpath):
		os.makedirs(plotpath)

	# n_layers = 4
	# hidden_ls = [1024,512, 256, 128]
	# # size_ls = [8,12,20,32,48,64,96,128]
	# #			 5,6 ,8 ,9 ,10,10,14,18 
	# n_epochs = 80
	
	# hidden_ls = [1024,512, 256, 64]
	hidden_ls = [2056,1024,512,358,256]
	bottle_neck = hidden_ls[-1]
	n_layers = len(hidden_ls)
	# size_ls = [6,8,12,20,32,48,64]
	#			 5,8 ,8,10,12,15,20,    #(total 80)
	size_ls = np.load(modelpath+'dae/0/size_each_epoch.npy')

	
	n_epochs = 130

	sae = load_model(modelpath+'sae/0/', 'sae', n_layers, hidden_ls, size_ls, n_epochs-1)
	dae = load_model(modelpath+'dae/0/','dae', n_layers, hidden_ls, size_ls, n_epochs-1)

	activation_dictionary = {}
	layer_name = 'bottle_neck'
	sae.encoder.encoder_3.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

	data = torch.randn(3,32, 32)
	data = data.unsqueeze(0)
	input = data.view(data.size(0), -1)
	input.requires_grad_(True)

	steps = 100               # perform 100 iterations
	unit = 0                  # flamingo class of Imagenet
	alpha = torch.tensor(100) # learning rate (step size)
	verbose = False           # print activation every step
	L2_Decay = True           # enable L2 decay regularizer
	Gaussian_Blur = False     # enable Gaussian regularizer
	Norm_Crop = False         # enable norm regularizer
	Contrib_Crop = False      # enable contribution regularizer

	sae_rf_ls = []
	for i in range(bottle_neck):
		output = amu.act_max(network=sae,
						input=input,
						layer_activation=activation_dictionary,
						layer_name=layer_name,
						unit=i,
						steps=steps,
						alpha=alpha,
						verbose=verbose,
						L2_Decay=L2_Decay,
						Gaussian_Blur=Gaussian_Blur,
						Norm_Crop=Norm_Crop,
						Contrib_Crop=Contrib_Crop,
						)
		sae_rf_ls.append(output.detach().numpy())
	np.save(modelpath + 'sae_maxact.npy', sae_rf_ls)


	fig = plt.figure(figsize=(30,30))
	for i in range(bottle_neck):
		plt.subplot(16,16,i+1)
		if i == 0:
			plt.title('sae')
		else:
			plt.title(str(i+1))
		plt.imshow(normalize_image(np.transpose(sae_rf_ls[i].reshape(3,32,32), (1,2,0))))
	plt.savefig(plotpath + 'sae_maxact_norm.png',bbox_inches='tight')
	plt.close(fig)

	fig = plt.figure(figsize=(30,30))
	for i in range(bottle_neck):
		plt.subplot(16,16,i+1)
		if i == 0:
			plt.title('sae')
		else:
			plt.title(str(i+1))
		plt.imshow(np.transpose(sae_rf_ls[i].reshape(3,32,32), (1,2,0)))
	plt.savefig(plotpath + 'sae_maxact.png',bbox_inches='tight')
	plt.close(fig)

	activation_dictionary = {}
	layer_name = 'bottle_neck'
	dae.encoder.encoder_3.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

	data = torch.randn(3,32, 32)
	data = data.unsqueeze(0)
	input = data.view(data.size(0), -1)
	input.requires_grad_(True)
	print(input.shape)

	dae_rf_ls = []
	for i in range(bottle_neck):
		output = amu.act_max(network=dae,
						input=input,
						layer_activation=activation_dictionary,
						layer_name=layer_name,
						unit=i,
						steps=steps,
						alpha=alpha,
						verbose=verbose,
						L2_Decay=L2_Decay,
						Gaussian_Blur=Gaussian_Blur,
						Norm_Crop=Norm_Crop,
						Contrib_Crop=Contrib_Crop,
						)
		dae_rf_ls.append(output.detach().numpy())
	np.save(modelpath + 'dae_maxact.npy', dae_rf_ls)

	fig = plt.figure(figsize=(30,30))
	for i in range(bottle_neck):
		plt.subplot(16,16,i+1)
		if i == 0:
			plt.title('dae')
		else:
			plt.title(str(i+1))
		plt.imshow(normalize_image(np.transpose(dae_rf_ls[i].reshape(3,32,32), (1,2,0))))
	plt.savefig(plotpath + 'dae_maxact_norm.png',bbox_inches='tight')
	plt.close(fig)

	fig = plt.figure(figsize=(30,30))
	for i in range(bottle_neck):
		plt.subplot(16,16,i+1)
		if i == 0:
			plt.title('dae')
		else:
			plt.title(str(i+1))
		plt.imshow(np.transpose(dae_rf_ls[i].reshape(3,32,32), (1,2,0)))
	plt.savefig(plotpath + 'dae_maxact.png',bbox_inches='tight')
	plt.close(fig)
