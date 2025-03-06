import os

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import NonLinearAutoencoder
from solver import train_vali_all_epochs, nl_dev_train_vali_all_epochs, test


def train_sae(save_path,n_layers,hidden_ls,n_epochs=10):
	sae_model = NonLinearAutoencoder(n_input=3072, n_hidden_ls=hidden_ls, n_layers=n_layers)
	sae_optimizer = torch.optim.SGD(sae_model.parameters(), lr=0.1, momentum=0.9)

	sae_train_loss, sae_validation_loss = train_vali_all_epochs(
		sae_model, 
		train_loader, 
		validation_loader, 
		sae_optimizer, 
		n_epochs=n_epochs, 
		device=device, 
		save_path=save_path+'/'
		)

	return sae_model, sae_train_loss, sae_validation_loss


def train_dae(save_path,n_layers,hidden_ls,size_ls,n_epochs=10):
	dae_model = NonLinearAutoencoder(n_input=3072, n_hidden_ls=hidden_ls, n_layers=n_layers)
	dae_optimizer = torch.optim.SGD(dae_model.parameters(), lr=0.1, momentum=0.9)

	size_ls = size_ls

	manner = 'cell_division'

	dae_train_loss, dae_validation_loss = nl_dev_train_vali_all_epochs(
		dae_model, 
		size_ls,
		manner,
		train_loader,
		validation_loader, 
		dae_optimizer, 
		n_epochs=n_epochs,
		device=device, 
		save_path=save_path+'/'
		)

	return dae_model, dae_train_loss, dae_validation_loss



def load_model(model_path,model_type, n_layers,n_hidden_ls,size_ls, epoch):
	n_input = 3072
	sae_n_hidden_ls = n_hidden_ls

	dae_n_hidden_ls = n_hidden_ls

	hidden_layers = sae_n_hidden_ls if model_type == 'sae' else dae_n_hidden_ls
	model = NonLinearAutoencoder(n_input, hidden_layers, n_layers)

	weights = torch.load(f"{model_path}/model_weights_epoch{epoch}.pth")
	model.load_state_dict(weights)
	return model


if __name__ == '__main__':
	print('cuda',torch.cuda.is_available())
	device = torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	run_id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

	# CIFAR10 normalization (RGB channels)
	transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors and scale to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
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

	num_models = 1
	for i in range(num_models):
		# save_path looks like: /home/david/cifar_models/model_type/iteration/'
		base_dir = os.path.join(os.getenv("HOME"), "cifar_models", run_id)
		
		sae_path = os.path.join(base_dir, "sae", str(i))
		dae_path = os.path.join(base_dir, "dae", str(i))
		plot_path = os.path.join(base_dir, "plots")
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)
		if not os.path.exists(sae_path):
			os.makedirs(sae_path)
		if not os.path.exists(dae_path):
			os.makedirs(dae_path)

		
		# hidden_ls = [1024,512, 256, 64]
		hidden_ls = [2056,1024,512,358,256]
		n_layers = len(hidden_ls)
		# size_ls = [ 6,10,16,24,32,48,64,86,128,200,256]
		#			  5,8 , 8, 8,10,10,12,12, 15,20 ,22 #(total)    
		size_ls = [6,6,6,6,6,
			 	   10,10,10,10,10,10,10,10,
				   16,16,16,16,16,16,16,16,
			 	   24,24,24,24,24,24,24,24, 
				   32,32,32,32,32,32,32,32,32,32,
				   48,48,48,48,48,48,48,48,48,48,
				   64,64,64,64,64,64,64,64,64,64,64,64,
				   86,86,86,86,86,86,86,86,86,86,86,86,
				   128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
				   200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,
				   256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256]
		
		n_epochs = 130
		_, sae_train_loss, sae_validation_loss = train_sae(sae_path,n_layers,hidden_ls,n_epochs)
		_, dae_train_loss, dae_validation_loss = train_dae(dae_path,n_layers,hidden_ls,size_ls,n_epochs)

		## plot loss
		fig = plt.figure(figsize=(8, 4))
		ax = fig.add_subplot(121)
		ax.plot(sae_train_loss, label='SAE train')
		ax.plot(dae_train_loss, label='DAE train')
		ax.legend()
		ax.set_title('Train loss')
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Loss')

		ax = fig.add_subplot(122)
		ax.plot(sae_validation_loss, label='SAE validation')
		ax.plot(dae_validation_loss, label='DAE validation')
		ax.legend()
		ax.set_title('Validation loss')
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Loss')

		plt.tight_layout()
		plt.savefig(os.path.join(plot_path, "loss.png"))
		plt.close(fig)


		## load model
		sae_model = load_model(sae_path, 'sae', n_layers, hidden_ls, size_ls, n_epochs-1)
		dae_model = load_model(dae_path, 'dae', n_layers, hidden_ls, size_ls, n_epochs-1)

		## test
		sae_test_loss = test(sae_model, test_loader, device)
		dae_test_loss = test(dae_model, test_loader, device)
		print(f"SAE test loss: {sae_test_loss}")
		print(f"DAE test loss: {dae_test_loss}")

		## plot reconstructions
		sae_model.eval()
		images, labels = next(iter(train_loader))
		sample_img = images[0]  # Shape: [C, H, W]
		sample_label = labels[0]  # Scalar value
		# Flatten the image
		flat_sample_img = sample_img.view(-1)  # Shape: [C*H*W]
		_, reco_sample_img = sae_model(flat_sample_img)
		reco_sample_img = reco_sample_img.view(sample_img.size())  # Shape: [C, H, W]

		# Denormalize the image
		# mean = np.array([0.4914, 0.4822, 0.4465])
		# std = np.array([0.2470, 0.2435, 0.2616])
		reco_sample_img = (0.5 * reco_sample_img) + 0.5  # Denormalize
		reco_sample_img = reco_sample_img.detach().numpy().transpose(1, 2, 0)  # Convert to HxWxC
		# reco_sample_img = np.clip(reco_sample_img, 0, 1)  # Clip to valid range



		fig, axes = plt.subplots(2, 2, figsize=(10, 10))
		axes[0,0].imshow(sample_img.numpy().transpose(1, 2, 0))
		axes[0,0].set_title(f"original Image")
		axes[0,0].axis('off')
		axes[0,1].imshow(reco_sample_img)
		axes[0,1].set_title(f"sae Reconstructed Image")
		axes[0,1].axis('off')
		


		dae_model.eval()
		flat_sample_img = sample_img.view(-1)  # Shape: [C*H*W]
		_, reco_sample_img = dae_model(flat_sample_img)
		reco_sample_img = reco_sample_img.view(sample_img.size())  # Shape: [C, H, W]
		# Denormalize the image
		# mean = np.array([0.4914, 0.4822, 0.4465])
		# std = np.array([0.2470, 0.2435, 0.2616])
		reco_sample_img = (0.5 * reco_sample_img )+ 0.5  # Denormalize
		reco_sample_img = reco_sample_img.detach().numpy().transpose(1, 2, 0)  # Convert to HxWxC
		
		# reco_sample_img = np.clip(reco_sample_img, 0, 1)  # Clip to valid range

		
		axes[1,0].imshow(sample_img.numpy().transpose(1, 2, 0))
		axes[1,0].set_title(f"original Image")
		axes[1,0].axis('off')
		axes[1,1].imshow(reco_sample_img)
		axes[1,1].set_title(f"dae Reconstructed Image")
		axes[1,1].axis('off')

		plt.tight_layout()

		plt.savefig(os.path.join(plot_path, "reconstruction.png"))


	