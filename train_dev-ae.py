import numpy as np
import torch
import os
import argparse
import sys

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import ConvAutoencoder
from solver import dev_train_vali_all_epochs_conv

def get_dataloaders(batch_size=128, num_workers=6):
    """
    Standardized CIFAR10 dataloader creation.
    """
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
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, validation_loader

def main():
    parser = argparse.ArgumentParser(description='Train Dev-AE')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_train_runs', type=int, default=40, help='Number of independent training runs')
    parser.add_argument('--n_epochs', type=int, default=60, help='Total number of epochs')
    parser.add_argument('--manner', type=str, default='cell_division', choices=['naiv', 'cell_division'], 
                        help='Method for initializing new neurons') # not entirely if naiv is correctly
                        # implemented so just using cell_division bc it's the same with 70% increase
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.expanduser('~'), './aiii26/models/dev-ae'),
                        help='Directory to save models and losses')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    train_loader, validation_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # 60 epochs total: 6, 10, 17, 29, 50, 85, 128
    size_ls = [6] * 6 + [10] * 6 + [17] * 7 + [29] * 7 + [50] * 8 + [85] * 8 + [128] * 18
    
    for run_i in range(args.num_train_runs):
        print(f"\n--- Training Run {run_i + 1}/{args.num_train_runs} ---")
        
        # Setup run-specific save path
        run_save_path = os.path.join(args.save_dir, str(run_i))
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path)

        # Initialize model with starting size
        dae_model = ConvAutoencoder(latent_dim=size_ls[0]).to(device)
        dae_optimizer = torch.optim.SGD(dae_model.parameters(), lr=0.1)

        # Train
        t_loss, v_loss = dev_train_vali_all_epochs_conv(
            model=dae_model, 
            size_ls=size_ls,
            train_loader=train_loader,
            vali_loader=validation_loader, 
            optimizer=dae_optimizer, 
            n_epochs=args.n_epochs,
            device=device, 
            save_path=run_save_path + '/',
            manner=args.manner,
        )

        # Save losses
        np.save(os.path.join(run_save_path, 'train_loss.npy'), t_loss)
        np.save(os.path.join(run_save_path, 'val_loss.npy'), v_loss)
        
        print(f"Run {run_i} completed. Saved to {run_save_path}")

    print("\nAll training runs completed.")

if __name__ == "__main__":
    main()