import os
from pc_stability import analyze_pc_stability
from rf_computation import compute_rfs
from rf_stability import analyze_rf_stability
from rf_specificity import compute_rf_specificity
from hidden_layer_statistics import compute_hidden_layer_activation
from plot_hidden_layer_statistics import plot_hidden_layer_activation
from power_spectra import plot_power_spectra
from pc_noise import run_pc_noise_analysis
from bottleneck_activation import run_bottleneck_activation_analysis

def main():
    size_ls = [4, 4, 4, 4, 4, 10,
            10, 10, 10, 10, 16, 16,
            16, 16, 16, 16, 16, 24,
            24, 24, 24, 24, 24, 24, 
            32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32,]
    
    # analyze_pc_stability("sae", size_ls=None, num_models=10, num_epochs=60)
    # analyze_pc_stability("dae", size_ls=size_ls, num_models=10, num_epochs=60)

    # compute_rfs("sae", size_ls=None, num_models=1, num_epochs=10)
    # compute_rfs("dae", size_ls=size_ls, num_models=1, num_epochs=10)

    # analyze_rf_stability("sae", size_ls=None, num_models=1, num_epochs=60)
    # analyze_rf_stability("dae", size_ls=size_ls, num_models=1, num_epochs=60)

    # compute_rf_specificity("sae", num_models=1, size_ls=None, num_epochs=60)
    # compute_rf_specificity("dae", num_models=1, size_ls=size_ls, num_epochs=60)

    # compute_hidden_layer_activation("sae", num_models=1, num_epochs=3)
    # compute_hidden_layer_activation("dae", num_models=1, num_epochs=3)

    # plot_hidden_layer_activation("sae", 2)
    # plot_hidden_layer_activation("dae", 2)

    # neuron_groups = [6, 10, 16, 28, 90, 128]
    # plot_power_spectra('Results/sae_maxact.npy', 'Results/dae_maxact.npy', 1, 0, neuron_groups=neuron_groups)

    run_pc_noise_analysis(2)

    # run_bottleneck_activation_analysis(10)

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
