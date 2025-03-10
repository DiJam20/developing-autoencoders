"""
created by: Deyue Kong
created on: Feb 25th

"""
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
import act_max_util as amu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import ConvAutoencoder
from solver import train_vali_all_epochs_conv, dev_train_vali_all_epochs_conv,test_conv


def train_sae(save_path,latent_dim,n_epochs=10,device='cpu'):
	sae_model = ConvAutoencoder(latent_dim=latent_dim)
	sae_optimizer = torch.optim.SGD(sae_model.parameters(), lr=0.1, momentum=0.9)

	sae_train_loss, sae_validation_loss = train_vali_all_epochs_conv(
		sae_model, 
		train_loader, 
		validation_loader, 
		sae_optimizer, 
		n_epochs=n_epochs, 
		device=device, 
		save_path=save_path
		)

	return sae_model, sae_train_loss, sae_validation_loss

def train_dae(save_path,latent_dim,size_ls,n_epochs=10):
	dae_model = ConvAutoencoder(latent_dim=latent_dim)
	dae_optimizer = torch.optim.SGD(sae_model.parameters(), lr=0.1, momentum=0.9)

	size_ls = size_ls
	manner = 'cell_division'

	dae_train_loss, dae_validation_loss = dev_train_vali_all_epochs_conv(
		dae_model, 
		size_ls,
		train_loader,
		validation_loader, 
		dae_optimizer, 
		n_epochs=n_epochs,
		device=device, 
		save_path=save_path,
		manner=manner
		)

	return dae_model, dae_train_loss, dae_validation_loss

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


def load_model(model_path,size_ls, epoch):
	print(size_ls[-1])
	model = ConvAutoencoder(latent_dim=size_ls[-1])
	weights = torch.load(f"{model_path}/model_weights_epoch{epoch}.pth")
	model.load_state_dict(weights)
	return model

if __name__ == '__main__':
	print('cuda',torch.cuda.is_available())
	device = torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	run_id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

	save_path = '/home/kong/cifar_models/cnn/{}/'.format(run_id)
	plot_path = '/home/kong/cifar_plots/cnn/{}/'.format(run_id)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	# CIFAR10 normalization (RGB channels)
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
	])

	# Load CIFAR10 datasets
	cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	# Split training set into train/validation
	train_size = int(len(cifar_train) * 0.8)  # 40,000 samples
	validation_size = len(cifar_train) - train_size  # 10,000 samples
	cifar_train, cifar_val = torch.utils.data.random_split(cifar_train, [train_size, validation_size])

	# Create data loaders
	batch_size = 128
	train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=6)
	validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=6)
	test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=6)


	latent_dim = 128
	n_epochs = 60

	repeat = 5
	for rep in range(repeat):	
		rep_plot_path = plot_path + f'{rep}/'
		if not os.path.exists(rep_plot_path):
			os.makedirs(rep_plot_path)
		sae_model, sae_train_loss, sae_validation_loss = train_sae(save_path+'sae/{}/'.format(rep),latent_dim,n_epochs=n_epochs,device=device)
		np.save(save_path+'sae/{}/train_loss.npy'.format(rep),sae_train_loss)
		np.save(save_path+'sae/{}/vali_loss.npy'.format(rep),sae_validation_loss)
		
		# size_ls = [6,10,16,28,48,90,128]#,196,256]
		size_ls = [  6,   6,   6,   6,   6,   6,    # 6
					10,  10,  10,  10,  10,  10,    # 6
					16,  16,  16,  16,  16,  16,    # 6
					28,  28,  28,  28,  28,  28,    # 6
					48,  48,  48,  48,  48,  48,  48,  48, 48, # 9
					90,  90,  90,  90,  90,  90,  90,  90,  90,  90, #10
					128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
					128, 128, 128, 128, 128, 128, 128 # 17
					]
		dae_model, dae_train_loss, dae_validation_loss = train_dae(save_path+'dae/{}/'.format(rep),latent_dim,size_ls,n_epochs=n_epochs)
		np.save(save_path+'dae/{}/train_loss.npy'.format(rep),dae_train_loss)
		np.save(save_path+'dae/{}/vali_loss.npy'.format(rep),dae_validation_loss)

		fig = plt.figure(figsize=(10,5))
		plt.subplot(1,2,1)
		plt.plot(sae_train_loss,label='SAE Train Loss')
		plt.plot(dae_train_loss,label='DAE Train Loss')
		plt.legend()
		plt.title('Train Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')

		plt.subplot(1,2,2)
		plt.plot(sae_validation_loss,label='SAE Validation Loss')
		plt.plot(dae_validation_loss,label='DAE Validation Loss')
		plt.legend()
		plt.title('Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.savefig(rep_plot_path+'loss.png')


		# Test the models
		test_loss, decoded,test = test_conv(sae_model,test_loader)
		fig = plt.figure(figsize=(20,10))
		for i in range(10):
			plt.subplot(3,10,i+1)
			plt.imshow(np.transpose(test[i].numpy(),(1,2,0)))
			if i == 0:
				plt.title('Original')
			plt.axis('off')

			plt.subplot(3,10,i+11)
			plt.imshow(np.transpose(decoded[i].detach().numpy(),(1,2,0)))
			if i == 0:
				plt.title('SAE Reconstructed')
			plt.axis('off')

		
		dae_model = load_model(save_path+'dae/{}/'.format(rep),size_ls,n_epochs-1)
		test_loss, decoded,test = test_conv(dae_model,test_loader)
		
		for i in range(10):
			plt.subplot(3,10,i+21)
			plt.imshow(np.transpose(decoded[i].detach().numpy(),(1,2,0)))
			if i == 0:
				plt.title('DAE Reconstructed')
			plt.axis('off')
		plt.savefig(rep_plot_path+'test_reconstruction.png')

		### receptive fields
		activation_dictionary = {}
		layer_name = 'bottle_neck'
		sae_model.encoder[11].register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

		data = torch.randn(3,32, 32)
		input = data.unsqueeze(0)
		# input = data.view(data.size(0), -1)
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
		for i in range(latent_dim):
			output = amu.act_max(network=sae_model,
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
		np.save(save_path + 'sae/{}/'.format(rep)+ 'sae_maxact.npy', sae_rf_ls)


		fig = plt.figure(figsize=(30,30))
		for i in range(latent_dim):
			plt.subplot(16,16,i+1)
			if i == 0:
				plt.title('sae')
			else:
				plt.title(str(i+1))
			plt.imshow(np.transpose(sae_rf_ls[i].reshape(3,32,32), (1,2,0)))
		plt.savefig(rep_plot_path + 'sae_maxact.png',bbox_inches='tight')
		plt.close(fig)

		activation_dictionary = {}
		layer_name = 'bottle_neck'
		dae_model.encoder[11].register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

		data = torch.randn(3,32, 32)
		input = data.unsqueeze(0)
		# input = data.view(data.size(0), -1)
		input.requires_grad_(True)
		print(input.shape)

		dae_rf_ls = []
		for i in range(latent_dim):
			output = amu.act_max(network=dae_model,
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
		np.save(save_path +'sae/{}/'.format(rep)+ 'dae_maxact.npy', dae_rf_ls)

		fig = plt.figure(figsize=(30,30))
		for i in range(latent_dim):
			plt.subplot(16,16,i+1)
			if i == 0:
				plt.title('dae')
			else:
				plt.title(str(i+1))
			plt.imshow(np.transpose(dae_rf_ls[i].reshape(3,32,32), (1,2,0)))
		plt.savefig(rep_plot_path + 'dae_maxact.png',bbox_inches='tight')
		plt.close(fig)





	


