import numpy as np
import torch
import torch.optim as optim
import os
import argparse
import sys

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoder import ConvAutoencoder
from solver import train_vali_all_epochs_conv

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
    parser = argparse.ArgumentParser(description='Train Standard SAE (ConvAutoencoder)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_train_runs', type=int, default=40, help='Number of independent training runs')
    parser.add_argument('--n_epochs', type=int, default=60, help='Total number of epochs per run')
    parser.add_argument('--latent_dim', type=int, default=128, help='Size of the bottleneck layer')
    parser.add_argument('--save_dir', type=str, default='./aiii26/models/ae',
                        help='Directory to save models and losses')
    
    args = parser.parse_args()
    
    # Ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get Data
    train_loader, validation_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"\nStarting training for {args.num_train_runs} runs...")
    print(f"Latent Dim: {args.latent_dim}, Epochs: {args.n_epochs}")

    # Training Loop
    for run_i in range(args.num_train_runs):
        print(f"\n--- Training Run {run_i + 1}/{args.num_train_runs} ---")
        
        # Setup run-specific save path (e.g., ./aiii26/models/ae/0)
        run_save_path = os.path.join(args.save_dir, str(run_i))
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path)

        sae_model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        sae_optimizer = torch.optim.SGD(sae_model.parameters(), lr=0.1)

        # Train
        sae_train_loss, sae_validation_loss = train_vali_all_epochs_conv(
            model=sae_model, 
            train_loader=train_loader, 
            vali_loader=validation_loader, 
            optimizer=sae_optimizer, 
            n_epochs=args.n_epochs, 
            device=device, 
            save_path=run_save_path + '/'
        )

        # Save losses explicitly at the end of the run
        np.save(os.path.join(run_save_path, 'train_loss.npy'), sae_train_loss)
        np.save(os.path.join(run_save_path, 'val_loss.npy'), sae_validation_loss)
        
        print(f"Run {run_i} completed. Saved to {run_save_path}")

    print("\nAll training runs completed.")

if __name__ == "__main__":
    main()