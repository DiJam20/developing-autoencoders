import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from autoencoder import ConvAutoencoder
from solver import (
    train_vali_all_epochs_conv,
    dev_train_vali_all_epochs_conv,
    initialize_conv_sae_with_pca,
    train_vali_all_epochs_with_bottleneck_freeze,
    dev_train_vali_converge_conv,
)
from train_converging_dev_ae import average_schedule_durations as avg_schedule

def get_loaders(batch_size=128, num_workers=6):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(train_set, [int(len(train_set)*0.8), int(len(train_set)*0.2)])
    return (DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=int, default=40, help='Number of runs for AE, PCA-AE, Dev-AE')
    parser.add_argument('--conv_dev_runs', type=int, default=10, help='Number of runs for Conv-Dev-AE')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--save_root', default='aiii26/models')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, val_loader = get_loaders()
    
    # AE
    print(f"\n[1/4] Training {args.num_models} Standard AEs...")
    for i in range(args.num_models):
        path = os.path.join(args.save_root, 'ae', str(i))
        model = ConvAutoencoder(latent_dim=128).to(device)
        train_vali_all_epochs_conv(model, train_loader, val_loader, optim.SGD(model.parameters(), lr=0.1), 
                                   n_epochs=args.epochs, device=device, save_path=path+'/')

    # PCA-AE
    print(f"\n[2/4] Training {args.num_models} PCA-AEs...")
    for i in range(args.num_models):
        path = os.path.join(args.save_root, 'pca_ae', str(i))
        model, _ = initialize_conv_sae_with_pca(128, device, train_loader)
        train_vali_all_epochs_with_bottleneck_freeze(model, train_loader, val_loader, device, path+'/', 
                                                     total_epochs=args.epochs, freeze_epochs=20, lr=0.1)

    # Dev-AE
    print(f"\n[3/4] Training {args.num_models} Dev-AEs...")
    fixed_sched = [6]*6 + [10]*6 + [17]*7 + [29]*7 + [50]*8 + [85]*8 + [128]*18
    
    for i in range(args.num_models):
        path = os.path.join(args.save_root, 'dev_ae', str(i))
        model = ConvAutoencoder(latent_dim=fixed_sched[0]).to(device)
        dev_train_vali_all_epochs_conv(model, fixed_sched, train_loader, val_loader, optim.SGD(model.parameters(), lr=0.1), 
                                       n_epochs=args.epochs, device=device, save_path=path+'/', manner='cell_division')

    # Conv-Dev-AE
    start_sizes = [4, 6, 10, 16, 24]
    print(f"\n[4/4] Training Conv-Dev-AEs for starts: {start_sizes}...")
    
    for start in start_sizes:
        base_dir = os.path.join(args.save_root, 'converging_dev_ae', f'start_{start}')
        
        # Find Schedule (5 runs)
        milestones = []
        curr = start
        while curr < 128:
            curr = int(curr * 1.7)
            if curr > 128: curr = 128
            if not milestones or curr > milestones[-1]: milestones.append(curr)
            
        collected_epochs = []
        for j in range(5):
            _, _, per_epoch = dev_train_vali_converge_conv(milestones, 'naiv', train_loader, val_loader, 
                                                           device=device, save_path=os.path.join(base_dir, 'temp', f'search_{j}/'))
            collected_epochs.append(per_epoch)
        
        shutil.rmtree(os.path.join(base_dir, 'temp'), ignore_errors=True)
        
        final_sched = avg_schedule(collected_epochs, milestones, args.epochs)
        np.save(os.path.join(base_dir, 'averaged_schedule.npy'), final_sched)

        # Train Models (10 runs)
        for j in range(args.conv_dev_runs):
            path = os.path.join(base_dir, str(j))
            model = ConvAutoencoder(latent_dim=final_sched[0]).to(device)
            dev_train_vali_all_epochs_conv(model, final_sched, train_loader, val_loader, optim.SGD(model.parameters(), lr=0.1), 
                                           n_epochs=args.epochs, device=device, save_path=path+'/', manner='naiv')

if __name__ == "__main__":
    main()