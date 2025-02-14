## import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


## define the autoencoder
class LinearAutoencoder(nn.Module):
	## initialization
	def __init__(self,n_input,n_hidden_ls,n_layers):
		super(LinearAutoencoder,self).__init__()
		self.n_input = n_input
		self.n_hidden_ls = n_hidden_ls
		self.n_layers = n_layers
		self.encoder = nn.Sequential()
		self.decoder = nn.Sequential()

		for i in range(n_layers):
			if i == 0:
				self.encoder.add_module(f'encoder_{i+1}',nn.Linear(n_input,n_hidden_ls[i]))
			else:
				self.encoder.add_module(f'encoder_{i+1}',nn.Linear(n_hidden_ls[i-1],n_hidden_ls[i]))
				
		n_hidden_ls_reversed = n_hidden_ls[::-1]
		for j in range(n_layers):
			if j == n_layers-1:
				self.decoder.add_module(f'decoder_{j+1}',nn.Linear(n_hidden_ls_reversed[j],n_input))
			else:
				self.decoder.add_module(f'decoder_{j+1}',nn.Linear(n_hidden_ls_reversed[j],n_hidden_ls_reversed[j+1]))
		
	## forward propagation
	def forward(self,x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded
	
	def encode(self,x):
		encoded = self.encoder(x)
		return encoded
	
	def decode(self,x):
		decoded = self.decoder(x)
		return decoded
	
	def get_hyperparams(self):
		params = {
			'n_input': self.n_input,
			'n_hidden_ls': self.n_hidden_ls,
			'n_layers': self.n_layers
		}
		return params

class NonLinearAutoencoder(nn.Module):
	## initialization
	def __init__(self,n_input,n_hidden_ls,n_layers):
		super(NonLinearAutoencoder,self).__init__()
		self.n_input = n_input
		self.n_hidden_ls = n_hidden_ls
		self.n_layers = n_layers
		self.encoder = nn.Sequential()
		self.decoder = nn.Sequential()

		for i in range(n_layers):
			if i == 0:
				self.encoder.add_module(f'encoder_{i+1}',nn.Linear(n_input,n_hidden_ls[i]))
			else:
				self.encoder.add_module(f'encoder_{i+1}',nn.Linear(n_hidden_ls[i-1],n_hidden_ls[i]))
			self.encoder.add_module(f'activation_{i+1}', nn.ReLU())
			# If no ReLU wanted after before the bottleneck layer, comment out the following line
			# if i != 2:
			# 	self.encoder.add_module(f'activation_{i+1}', nn.ReLU())
			# Add dropout layer after each hidden layer, but no dropout on the bottleneck layer
			# if i < n_layers - 1:
			# 	self.encoder.add_module(f'dropout_{i+1}', nn.Dropout(p=0.2))
				
		n_hidden_ls_reversed = n_hidden_ls[::-1]
		for j in range(n_layers):
			if j == n_layers-1:
				self.decoder.add_module(f'decoder_{j+1}',nn.Linear(n_hidden_ls_reversed[j],n_input))
			else:
				self.decoder.add_module(f'decoder_{j+1}',nn.Linear(n_hidden_ls_reversed[j],n_hidden_ls_reversed[j+1]))
				self.decoder.add_module(f'activation_{j+1}', nn.ReLU())
				# self.decoder.add_module(f'dropout_{j+1}', nn.Dropout(p=0.2))
		
	## forward propagation
	def forward(self, x, return_activations=False):
		activations = {}
		
		def get_activation(name):
			def hook(model, input, output):
				activations[name] = output.detach()
			return hook
		
		# Register hooks for each layer
		handles = []
		for name, layer in self.encoder.named_children():
			handles.append(layer.register_forward_hook(get_activation(f'encoder_{name}')))
		for name, layer in self.decoder.named_children():
			handles.append(layer.register_forward_hook(get_activation(f'decoder_{name}')))
		
		# Forward pass
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		
		# Remove hooks
		for handle in handles:
			handle.remove()
		
		if return_activations:
			return encoded, decoded, activations
		return encoded, decoded
	
	def encode(self,x):
		encoded = self.encoder(x)
		return encoded
	
	def decode(self,x):
		decoded = self.decoder(x)
		return decoded
	
	def get_hyperparams(self):
		params = {
			'n_input': self.n_input,
			'n_hidden_ls': self.n_hidden_ls,
			'n_layers': self.n_layers
		}
		return params


class ConvAutoencoder(nn.Module):
	def __init__(self):
		super(ConvAutoencoder,self).__init__()
		self.encoder = nn.Sequential(
			# size: batch x 3 x 32 x 32
			#in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
			nn.Conv2d(3,16,3,stride=2,padding=1),
			nn.ReLU(True),
			# size: batch x 8 x 16 x 16
			nn.MaxPool2d(2,stride=2),
			# size: batch x 8 x 8 x 8
			nn.Conv2d(16,8,3,stride=1,padding=1),
			nn.ReLU(True),
			# size: batch x 8 x 8 x 8
			nn.MaxPool2d(2,stride=2)
			# size: batch x 8 x 4 x 4
		)
		## convtranspose2d output size = (input_size-1)*stride - 2*padding + kernel_size + output_padding
		## H_out ​= (H_in​−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1


		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8,16,3,stride=2),
			nn.ReLU(True),
			# size: batch x 16 x 9 x 9
			nn.ConvTranspose2d(16,8,5,stride=2,padding=2),
			nn.ReLU(True),
			# size: batch x 8 x 18 x 18
			nn.ConvTranspose2d(8,3,2,stride=2,padding=1),
			nn.Tanh()
			# size: batch x 3 x 32 x 32
		)
	def forward(self,x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded


class VAE(nn.Module):
	def __init__(self,n_input,n_hidden_ls,n_layers,n_latent):
		super(VAE,self).__init__()
		self.encoder = nn.Sequential()
		self.decoder = nn.Sequential()

		for i in range(n_layers):
			if i == 0:
				self.encoder.add_module('encoder_{}'.format(i+1),nn.Linear(n_input,n_hidden_ls[i]))
				self.decoder.add_module('decoder_{}'.format(i+1),nn.Linear(n_hidden_ls[n_layers-i-1],n_hidden_ls[n_layers-i-2]))
			elif i == n_layers-1:
				self.encoder.add_module('encoder_{}'.format(i+1),nn.Linear(n_hidden_ls[i-1],n_hidden_ls[i]))
				self.decoder.add_module('decoder_{}'.format(i+1),nn.Linear(n_hidden_ls[n_layers-i-1],n_input))
			else:
				self.encoder.add_module('encoder_{}'.format(i+1),nn.Linear(n_hidden_ls[i-1],n_hidden_ls[i]))
				self.decoder.add_module('decoder_{}'.format(i+1),nn.Linear(n_hidden_ls[n_layers-i-1],n_hidden_ls[n_layers-i-2]))
		
		self.fc1 = nn.Linear(n_hidden_ls[-1],n_latent)
		self.fc2 = nn.Linear(n_hidden_ls[-1],n_latent)
		self.fc3 = nn.Linear(n_latent,n_hidden_ls[-1])
		


	def encode(self,x):
		h1 = F.relu(self.encoder(x))
		return self.fc1(h1), self.fc2(h1)


	def reparameterize(self,mu,logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decode(self,z):
		h3 = F.relu(self.fc3(z))
		return self.decoder(h3)

	

	def forward(self,x):
		# print('debug x',x.shape)
		mu, logvar = self.encode(x)
		# print('debug mu, logvar',mu.shape,logvar.shape)
		z = self.reparameterize(mu,logvar)
		# print('debug z',z.shape)
		return self.decode(z), mu, logvar	

