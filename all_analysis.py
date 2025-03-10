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
    num_epoch = 60
    num_models = 1
    # size_ls = np.load('/home/kong/cifar_models/cnn/2025-03-05_13:59:06/dae/0/size_each_epoch.npy')
    size_ls = [  6,   6,   6,   6,   6,   6,    # 6
					10,  10,  10,  10,  10,  10,    # 6
					16,  16,  16,  16,  16,  16,    # 6
					28,  28,  28,  28,  28,  28,    # 6
					48,  48,  48,  48,  48,  48,  48,  48, 48, # 9
					90,  90,  90,  90,  90,  90,  90,  90,  90,  90, #10
					128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
					128, 128, 128, 128, 128, 128, 128 # 17
					]
    print(size_ls)
    
    analyze_pc_stability("sae", size_ls=None, num_models=num_models, num_epochs=num_epoch)
    analyze_pc_stability("dae", size_ls=size_ls, num_models=num_models, num_epochs=num_epoch)

    compute_rfs("sae", size_ls=None, num_models=num_models, num_epochs=num_epoch)
    compute_rfs("dae", size_ls=size_ls, num_models=num_models, num_epochs=num_epoch)

    analyze_rf_stability("sae", size_ls=None, num_models=num_models, num_epochs=num_epoch)
    analyze_rf_stability("dae", size_ls=size_ls, num_models=num_models, num_epochs=num_epoch)

    compute_rf_specificity("sae", num_models=num_models, size_ls=None, num_epochs=num_epoch)
    compute_rf_specificity("dae", num_models=num_models, size_ls=size_ls, num_epochs=num_epoch)

    # compute_hidden_layer_activation("sae", num_models=1, num_epochs=10, epoch=num_epoch)
    # compute_hidden_layer_activation("dae", num_models=1, num_epochs=num_epoch, epoch=num_epoch, size_ls=size_ls)

    # plot_power_spectra(num_models=1, epoch=num_epoch-1)

    run_pc_noise_analysis(num_models=num_models)

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
