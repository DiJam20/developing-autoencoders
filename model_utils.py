import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from autoencoder import NonLinearAutoencoder

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(len(mnist_train) * 0.8)
    validation_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, validation_size])

    batch_size = 128
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=6)
    validation_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=6)
    return train_loader, validation_loader, test_loader

def load_model(model_path, model_type, epoch, size_ls=None):
    n_input = 28*28
    n_layers = 3
    sae_n_hidden_ls = [512, 128, 32]

    if size_ls is None:
        size_ls = [4, 4, 4, 4, 4, 10,
                10, 10, 10, 10, 16, 16,
                16, 16, 16, 16, 16, 24,
                24, 24, 24, 24, 24, 24, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32]
    
    dae_n_hidden_ls = [512, 128, size_ls[epoch]]
    
    if model_type.lower() == 'sae':
        model = NonLinearAutoencoder(n_input, sae_n_hidden_ls, n_layers)
    else:
        model = NonLinearAutoencoder(n_input, dae_n_hidden_ls, n_layers)
    device = torch.device("cpu")
    weights = torch.load(
        f"{model_path}/model_weights_epoch{epoch}.pth", 
        map_location=device, 
        weights_only=True
    )
    model.load_state_dict(weights)
    return model

# cos(theta) = (pc1_epoch1 · pc1_epoch2) / (||pc1_epoch1|| * ||pc1_epoch2||)
def cosine_angle_between_pcs(pc_a, pc_b):
    numerator = np.dot(pc_a, pc_b)
    denominator = np.linalg.norm(pc_a) * np.linalg.norm(pc_b)

    if denominator == 0:
        return np.nan

    cos_value = np.clip(numerator / denominator, -1.0, 1.0)
    angle = np.arccos(cos_value) * 180 / np.pi
    
    return min(angle, 180 - angle)