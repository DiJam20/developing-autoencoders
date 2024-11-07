import numpy as np
import matplotlib.pyplot as plt
import torch
# from nn_interpretability.interpretation.am.general_am import ActivationMaximization

def get_RF_linear(model, data_loader,device='cpu'):
	params = model.get_hyperparams()
	n_hidden = params['n_hidden_ls'][-1]
	input_shape = data_loader.data.shape[1:]
	rf_shape = (n_hidden, np.prod(input_shape))
	receptive_field = torch.zeros(rf_shape, device=device) # (Hidden size, Input size)
	with torch.no_grad():
		for data, _ in data_loader:
			input = data.view(data.size(0), -1).to(device)
			activation = model.encode(input).to(device) # (Batch, Hidden size)
			prod = torch.einsum('bh,bi->bhi', activation, input).to(device)
			receptive_field += torch.mean(prod.to(device),dim=0).squeeze()
	receptive_field /= len(data_loader)
	return receptive_field


	