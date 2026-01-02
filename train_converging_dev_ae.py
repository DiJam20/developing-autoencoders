import numpy as np
import torch
import torch.optim as optim
import os
import shutil
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from solver import dev_train_vali_converge_conv, dev_train_vali_all_epochs_conv
from autoencoder import ConvAutoencoder


def get_dataloaders(batch_size=128, num_workers=6):
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

def generate_target_sizes(start_size, end_size, growth_factor):
    """
    Generates the discrete list of bottleneck sizes.
    Example: 4 -> 7 -> 12 -> ... -> 128
    """
    sizes = [start_size]
    current = start_size
    while current < end_size:
        current = int(current * (1 + growth_factor))
        if current >= end_size:
            current = end_size
        sizes.append(current)
    return sizes

def average_schedule_durations(collected_epoch_lists, milestone_sizes, target_total_epochs=60):
    """
    1. Counts how many epochs were spent at each size for every run.
    2. Averages these durations.
    3. Reconstructs a schedule of exactly target_total_epochs.
    """
    # Dictionary to hold list of durations for each specific size
    # e.g., {4: [5, 6, 4], 7: [2, 3, 2], ...}
    duration_map = {s: [] for s in milestone_sizes}

    for run_schedule in collected_epoch_lists:
        # Count occurrences of each size in this specific run
        counts = Counter(run_schedule)
        
        for size in milestone_sizes:
            # If a size wasn't reached in a run (e.g. run crashed or cut short), duration is nan
            duration_map[size].append(counts.get(size, np.nan))

    # Construct the averaged schedule
    final_schedule = []
    
    print("\n--- Averaging Statistics ---")
    for size in milestone_sizes:
        durations = duration_map[size]
        avg_duration = np.nanmean(durations)

        if np.isnan(avg_duration):
            print(f"Size {size}: Never reached in any run. Stopping schedule construction.")
            break

        # Round to nearest integer
        final_duration = int(np.round(avg_duration))
        
        # Ensure we don't skip a step entirely unless avg is truly 0 (which shouldn't happen if it ran)
        if final_duration == 0 and avg_duration > 0:
            final_duration = 1
            
        print(f"Size {size}: Durations {durations} -> Avg {avg_duration:.2f} -> Final {final_duration} epochs")
        
        final_schedule.extend([size] * final_duration)

    # Enforce exactly 60 epochs
    current_len = len(final_schedule)
    
    if current_len > target_total_epochs:
        print(f"Resulting schedule length {current_len} > {target_total_epochs}. Cutting end.")
        final_schedule = final_schedule[:target_total_epochs]
    elif current_len < target_total_epochs:
        diff = target_total_epochs - current_len
        last_val = final_schedule[-1] if final_schedule else milestone_sizes[0]
        print(f"Resulting schedule length {current_len} < {target_total_epochs}. Filling {diff} epochs with size {last_val}.")
        final_schedule.extend([last_val] * diff)
    else:
        print("Schedule length matches target exactly.")

    return np.array(final_schedule, dtype=int)


def main():
    parser = argparse.ArgumentParser(description='Automated Schedule Finder (Duration Averaging)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
    parser.add_argument('--start_neurons', type=int, default=4, help='Starting bottleneck size')
    parser.add_argument('--end_neurons', type=int, default=128, help='Final bottleneck size (may not get reached if convergence was too slow)')
    parser.add_argument('--growth_rate', type=float, default=0.70, help='Growth rate (0.70 = 70%)')
    parser.add_argument('--target_epochs', type=int, default=60, help='Fixed epoch length for final schedule')
    parser.add_argument('--num_search_runs', type=int, default=5, help='Number of runs to average schedule')
    parser.add_argument('--num_train_runs', type=int, default=10, help='Number of evaluation runs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temp_dir', type=str, default='./aiii26/models/temp_converging_dae')
    parser.add_argument('--save_dir', type=str, default='./aiii26/models/converging_dev-ae')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    train_loader, vali_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Find num_runs number of size_ls schedules
    target_sizes = generate_target_sizes(args.start_neurons, args.end_neurons, args.growth_rate)
    print(f"\nSchedule Search ({args.num_search_runs} runs)")
    print(f"Growth Steps: {target_sizes}")
    
    # Store the full epoch-by-epoch size list for each run
    collected_epoch_lists = []
    
    for run_i in range(args.num_search_runs):
        print(f"\n--- Search Run {run_i + 1}/{args.num_search_runs} ---")
        
        _, _, per_epoch_sizes = dev_train_vali_converge_conv(
            size_ls=target_sizes,
            manner='naiv', 
            train_loader=train_loader,
            vali_loader=vali_loader,
            device=device,
            save_path=os.path.join(args.temp_dir, f'{run_i}/'),
            max_epochs=args.target_epochs
        )
        
        collected_epoch_lists.append(per_epoch_sizes)
        print(f"Run {run_i+1} completed. Total epochs: {len(per_epoch_sizes)}")


    # Average the collected schedules
    print(f"\nComputing Averaged Schedule based on Durations")
    
    final_schedule = average_schedule_durations(
        collected_epoch_lists, 
        target_sizes, 
        target_total_epochs=args.target_epochs
    )
    
    schedule_path = os.path.join(args.save_dir, 'averaged_schedule.npy')
    np.save(schedule_path, final_schedule)
    
    print(f"\nFINAL AVERAGED SCHEDULE (Length {len(final_schedule)}):")
    print(final_schedule)
    print(f"Saved to {schedule_path}")
    
    # Clean up temporary directory
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
        print(f"\nCleaned up temporary directory: {args.temp_dir}")

    # Train and save final models using the averaged schedule
    print(f"\nTraining of {args.num_train_runs} Final Models")
    
    save_dir = os.path.join(args.save_dir, f'{args.start_neurons}_start_neurons')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for run_i in range(args.num_train_runs):
        print(f"\n--- Final Model Training Run {run_i + 1}/{args.num_train_runs} ---")
        
        # Create a specific directory for this run
        run_save_path = os.path.join(save_dir, f'{run_i}')
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path)

        # Initialize base model
        conv_dae = ConvAutoencoder(latent_dim=final_schedule[0]).to(device)
        optimizer = optim.SGD(conv_dae.parameters(), lr=0.1) 

        # Train using the fixed averaged schedule
        t_loss, v_loss = dev_train_vali_all_epochs_conv(
            model=conv_dae,
            size_ls=final_schedule, 
            train_loader=train_loader,
            vali_loader=vali_loader,
            optimizer=optimizer,
            n_epochs=args.target_epochs,
            device=device,
            save_path=run_save_path + '/',
            manner='naiv'
        )
                
        np.save(os.path.join(run_save_path, 'train_loss.npy'), t_loss)
        np.save(os.path.join(run_save_path, 'val_loss.npy'), v_loss)
        
    print("\nAll models trained and saved.")
    print(f"Check directory: {save_dir}")

if __name__ == "__main__":
    main()