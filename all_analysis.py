import os
import numpy as np
from analysis_cifar.pc_stability import analyze_pc_stability
from analysis_cifar.rf_computation import compute_rfs
from analysis_cifar.rf_stability import analyze_rf_stability
from analysis_cifar.rf_specificity import compute_rf_specificity
from analysis_cifar.neuron_activation import compute_hidden_layer_activation
from analysis_cifar.power_spectra import plot_power_spectra
from analysis_cifar.pc_noise import run_pc_noise_analysis

def main():
    num_epoch = 50
    size_ls = np.load('/home/kong/cifar_models/cnn/2025-03-05_13:59:06/dae/0/size_each_epoch.npy')
    print(size_ls)
    
    # analyze_pc_stability("sae", size_ls=None, num_models=1, num_epochs=num_epoch)
    # analyze_pc_stability("dae", size_ls=size_ls, num_models=1, num_epochs=num_epoch)

    # compute_rfs("sae", size_ls=None, num_models=1, num_epochs=num_epoch)
    # compute_rfs("dae", size_ls=size_ls, num_models=1, num_epochs=num_epoch)

    # analyze_rf_stability("sae", size_ls=None, num_models=1, num_epochs=num_epoch)
    # analyze_rf_stability("dae", size_ls=size_ls, num_models=1, num_epochs=num_epoch)

    compute_rf_specificity("sae", num_models=1, size_ls=None, num_epochs=num_epoch)
    compute_rf_specificity("dae", num_models=1, size_ls=size_ls, num_epochs=num_epoch)

    compute_hidden_layer_activation("sae", num_models=1, num_epochs=10, epoch=9)
    compute_hidden_layer_activation("dae", num_models=1, num_epochs=num_epoch, epoch=9, size_ls=size_ls)

    plot_power_spectra(1, 59)

    run_pc_noise_analysis(2)

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
