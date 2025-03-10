import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import ConvAutoencoder
from solver import test_conv

def load_loss(run_id,nreps):
	all_loss = {
		'dae':{
			'train':[],
			'validation':[]
		},
		'sae':{
			'train':[],
			'validation':[]
		},
	}
	model_types = ['dae','sae']
	for model_type in model_types:
		for rep in range(nreps):
			datapath = '/home/kong/cifar_models/cnn/{}/{}/{}/'.format(run_id,model_type,rep)
			train_loss = np.load(datapath+'train_loss.npy')
			validation_loss = np.load(datapath+'vali_loss.npy')
			all_loss[model_type]['train'].append(train_loss)
			all_loss[model_type]['validation'].append(validation_loss)
	return all_loss

def load_model(model_path,size_ls, epoch):
	print(size_ls[-1])
	model = ConvAutoencoder(latent_dim=size_ls[-1])
	weights = torch.load(f"{model_path}/model_weights_epoch{epoch}.pth")
	model.load_state_dict(weights)
	return model


def get_resonstructions(run_id,n_epochs,size_ls,rep=0):
	# CIFAR10 normalization (RGB channels)
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
	])
	# Load CIFAR10 datasets
	cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	# Create data loaders
	batch_size = 128
	test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=6)

	reconstructions = {
		'dae':[],
		'sae':[],
		'original':[]
	}
	model_types = ['dae','sae']
	for model_type in model_types:
		model_path = '/home/kong/cifar_models/cnn/{}/{}/{}/'.format(run_id,model_type,rep)
		model = load_model(model_path,size_ls,n_epochs-1)
		test_loss, decoded,test = test_conv(model,test_loader)
		reconstructions[model_type] = decoded
		reconstructions['original'] = test
	return reconstructions

	



