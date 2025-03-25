import os
from pc_stability import analyze_pc_stability
from rf_computation import compute_rfs
from rf_stability import analyze_rf_stability
from rf_specificity import compute_rf_specificity
from hidden_layer_statistics import compute_hidden_layer_activation
from plot_hidden_layer_statistics import plot_hidden_layer_activation
from power_spectra import plot_power_spectra
from pc_noise import *
from bottleneck_activation import run_bottleneck_activation_analysis
from encoding_noise import run_encoding_noise_analysis
from classification_accuracy import run_all_classification_analyses
from dimensionality_development import run_dimensionality_analysis

def main():
    base_path = '/home/david/'

    os.makedirs("Results/figures/png", exist_ok=True)
    os.makedirs("Results/figures/svg", exist_ok=True)

    mnist_size_ls = [4, 4, 4, 4, 4, 10,
            10, 10, 10, 10, 16, 16,
            16, 16, 16, 16, 16, 24,
            24, 24, 24, 24, 24, 24, 
            32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32, 
            32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32,]
    
    cifar_size_ls = [6,   6,   6,   6,   6,   6,    # 6
					10,  10,  10,  10,  10,  10,    # 6
					16,  16,  16,  16,  16,  16,    # 6
					28,  28,  28,  28,  28,  28,    # 6
					48,  48,  48,  48,  48,  48,  48,  48, 48, # 9
					90,  90,  90,  90,  90,  90,  90,  90,  90,  90, #10
					128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
					128, 128, 128, 128, 128, 128, 128 # 17
					]
    
    # analyze_pc_stability("sae", dataset='mnist', compare_final_epoch=True, size_ls=None, num_models=10, num_epochs=60)
    # analyze_pc_stability("dae", dataset='mnist', compare_final_epoch=True, size_ls=mnist_size_ls, num_models=10, num_epochs=60)
    # analyze_pc_stability("sae", dataset='cifar', compare_final_epoch=True, size_ls=None, num_models=1, num_epochs=20)
    # analyze_pc_stability("dae", dataset='cifar', compare_final_epoch=True, size_ls=cifar_size_ls, num_models=1, num_epochs=20)

    # compute_rfs("sae", dataset='mnist', size_ls=None, num_models=2, num_epochs=60)
    # compute_rfs("dae", dataset='mnist', size_ls=mnist_size_ls, num_models=2, num_epochs=60)
    # compute_rfs("sae", dataset='cifar', size_ls=None, num_models=1, num_epochs=60)
    # compute_rfs("dae", dataset='cifar', size_ls=cifar_size_ls, num_models=2, num_epochs=60)

    # analyze_rf_stability("sae", dataset='mnist', compare_final_epoch=True, size_ls=None, num_models=2, num_epochs=60)
    # analyze_rf_stability("dae", dataset='mnist', compare_final_epoch=True, size_ls=mnist_size_ls, num_models=2, num_epochs=60)
    # analyze_rf_stability("sae", dataset='cifar', compare_final_epoch=True, size_ls=None, num_models=1, num_epochs=60)
    # analyze_rf_stability("dae", dataset='cifar', compare_final_epoch=True, size_ls=cifar_size_ls, num_models=2, num_epochs=60)

    # compute_rf_specificity("sae", dataset='cifar', num_models=1, size_ls=None, num_epochs=60)
    # compute_rf_specificity("dae", dataset='cifar', num_models=1, size_ls=mnist_size_ls, num_epochs=60)

    # compute_hidden_layer_activation('sae', 'nonlinear', num_models=5, num_epochs=1, epoch=59)
    # compute_hidden_layer_activation('dae', 'nonlinear', num_models=5, num_epochs=1, epoch=59)
    # compute_hidden_layer_activation('sae', 'conv', num_models=5, num_epochs=1, epoch=59)
    # compute_hidden_layer_activation('dae', 'conv', num_models=5, num_epochs=1, epoch=59)

    # plot_hidden_layer_activation(model_arch='nonlinear')
    # plot_hidden_layer_activation(model_arch='conv')

    # neuron_groups = [6, 10, 16, 28, 90, 128]
    # plot_power_spectra('Results/sae_maxact.npy', 'Results/dae_maxact.npy', 1, 0, neuron_groups=neuron_groups)


    # run_bottleneck_activation_analysis(40, 'mnist', base_path)
    # run_bottleneck_activation_analysis(10, 'cifar', base_path)


    # run_encoding_noise_analysis(2, [4, 10, 16, 24, 32], 'mnist')
    # run_encoding_noise_analysis(5, [6, 10, 16, 28, 48, 90, 128], 'cifar')

    mnist_manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    cifar_manipulated_neurons = [(0, 6), (6, 10), (10, 16), (16, 28), (28, 48), (48, 90), (90, 128)]

    # run_all_pc_analyses(40, "mnist", base_path, manipulated_neurons=mnist_manipulated)
    # run_all_pc_analyses(10, "cifar", base_path, manipulated_neurons=cifar_manipulated)

    # run_all_classification_analyses(1)

    # run_dimensionality_analysis('mnist', mnist_size_ls, num_models=2, num_epochs=30, base_path=base_path)
    # run_dimensionality_analysis('cifar', cifar_size_ls, num_models=2, num_epochs=60, base_path=base_path)

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
