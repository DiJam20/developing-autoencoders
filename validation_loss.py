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
            train_losses = np.load(file_path + 'train_loss.npy')
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


def get_vali_loss_per_epoch(model_type, dataset, num_models = 10, base_path = '/home/david/'):
    model_losses = []
    
    for model_id in range(num_models):
        file_path = f'{base_path}{dataset}_models/{model_type}/{model_id}/'
        train_losses = np.load(file_path + 'vali_loss.npy')

        # Convert MSE to RMSE
        rmse_normalized = np.sqrt(train_losses)
        # Denormalize RMSE
        rmse_original = rmse_normalized * 0.3081
        # Convert back to MSE
        train_losses = rmse_original ** 2

        model_losses.append(train_losses)
    
    model_losses_array = np.array(model_losses)

    return model_losses_array


def plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, sae_vali_loss, dae_vali_loss, dataset):
    sae_train_mean = np.mean(sae_train_loss, axis=0)
    sae_train_std = np.std(sae_train_loss, axis=0)
    dae_train_mean = np.mean(dae_train_loss, axis=0)
    dae_train_std = np.std(dae_train_loss, axis=0)

    if dataset == 'cifar':
        sae_vali_mean = np.mean(sae_vali_loss, axis=0)
        sae_vali_std = np.std(sae_vali_loss, axis=0)
        dae_vali_mean = np.mean(dae_vali_loss, axis=0)
        dae_vali_std = np.std(dae_vali_loss, axis=0)

    plt.rc('font', size=16)
    plt.figure(figsize=(6, 4), dpi=300)
    
    sae_train, = plt.plot(sae_train_mean, label='AE Train Loss', color='#1a7adb', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     sae_train_mean - sae_train_std, 
                     sae_train_mean + sae_train_std, 
                     color='#1a7adb', alpha=0.2)
    
    dae_train, = plt.plot(dae_train_mean, label='Dev-AE Train Loss', color='#e82817', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     dae_train_mean - dae_train_std, 
                     dae_train_mean + dae_train_std, 
                     color='#e82817', alpha=0.2)
    
    if dataset == 'cifar':
        sae_vali, = plt.plot(sae_vali_mean, label='AE Vali Loss', color='#1a7adb', linewidth=2, linestyle='--')
        plt.fill_between(range(len(sae_vali_loss[0])), 
                        sae_vali_mean - sae_vali_std, 
                        sae_vali_mean + sae_vali_std, 
                        color='#1a7adb', alpha=0.2)
        
        dae_vali, = plt.plot(dae_vali_mean, label='Dev-AE Vali Loss', color='#e82817', linewidth=2, linestyle='--')
        plt.fill_between(range(len(sae_vali_loss[0])), 
                        dae_vali_mean - dae_vali_std, 
                        dae_vali_mean + dae_vali_std, 
                        color='#e82817', alpha=0.2)
        
    plt.xticks([0, 29, 59], [1, 30, 60])
    plt.yticks([0, 0.1])
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curves', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    first_legend = plt.legend([sae_train, dae_train], ['AE', 'Dev-AE'], 
                             bbox_to_anchor=(1.0, 1.1),
                             loc='upper right', 
                             title='Train Loss')
    ax.add_artist(first_legend)
    
    if dataset == 'cifar':
        plt.legend([sae_vali, dae_vali], ['AE', 'Dev-AE'], 
                  bbox_to_anchor=(1.0, 0.7),
                  loc='upper right', 
                  title='Validation Loss')

    plt.tight_layout()

    plt.savefig(f'Results/figures/png/{dataset}_accuracy_over_epochs.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_accuracy_over_epochs.svg', bbox_inches='tight')
    plt.close()


def create_plots():
    sae_train_loss = get_train_loss_per_epoch('sae', 'mnist')
    dae_train_loss = get_train_loss_per_epoch('dae', 'mnist')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, None, None, 'mnist')

    sae_train_loss = get_train_loss_per_epoch('sae', 'cifar')
    dae_train_loss = get_train_loss_per_epoch('dae', 'cifar')
    sae_vali_loss = get_vali_loss_per_epoch('sae', 'cifar')
    dae_vali_loss = get_vali_loss_per_epoch('dae', 'cifar')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, sae_vali_loss, dae_vali_loss, 'cifar')