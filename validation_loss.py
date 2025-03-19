import numpy as np
import matplotlib.pyplot as plt
from solver import *


def get_train_loss_per_epoch(model_type, dataset, num_models = 10, base_path = '/home/david/'):
    model_losses = []
    
    for model_id in range(num_models):
        file_path = f'{base_path}{dataset}_models/{model_type}/{model_id}/'
        if dataset == 'mnist':
            train_losses = np.load(file_path + 'all_train_losses.npy')
        elif dataset == 'cifar':
            train_losses = np.load(file_path + 'vali_loss.npy')
        if dataset == 'mnist':
            train_losses = np.mean(train_losses, axis=1)

        # Convert MSE to RMSE
        rmse_normalized = np.sqrt(train_losses)
        # Denormalize RMSE
        rmse_original = rmse_normalized * 0.3081
        # Convert back to MSE
        train_losses = rmse_original ** 2

        model_losses.append(train_losses)
    
    model_losses_array = np.array(model_losses)

    return model_losses_array


def plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, dataset):
    sae_mean = np.mean(sae_train_loss, axis=0)
    sae_std = np.std(sae_train_loss, axis=0)
    dae_mean = np.mean(dae_train_loss, axis=0)
    dae_std = np.std(dae_train_loss, axis=0)

    plt.rc('font', size=16)
    plt.figure(figsize=(6, 4), dpi=300)
    
    plt.plot(sae_mean, label='AE', color='#1a7adb', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     sae_mean - sae_std, 
                     sae_mean + sae_std, 
                     color='#1a7adb', alpha=0.2)
    
    plt.plot(dae_mean, label='DevAE', color='#e82817', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     dae_mean - dae_std, 
                     dae_mean + dae_std, 
                     color='#e82817', alpha=0.2)
        
    plt.xticks([0, 29, 59], [1, 30, 60])
    plt.yticks([0, 0.1])
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Validation Loss Curve', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'Results/figures/png/{dataset}_accuracy_over_epochs.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_accuracy_over_epochs.svg', bbox_inches='tight')
    plt.close()


def create_plots():
    sae_train_loss = get_train_loss_per_epoch('sae', 'mnist')
    dae_train_loss = get_train_loss_per_epoch('dae', 'mnist')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, 'mnist')

    sae_train_loss = get_train_loss_per_epoch('sae', 'cifar')
    dae_train_loss = get_train_loss_per_epoch('dae', 'cifar')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, 'cifar')


create_plots()