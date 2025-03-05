import os
from pc_stability import analyze_pc_stability
from rf_computation import compute_rfs
from rf_stability import analyze_rf_stability
from rf_specificity import compute_rf_specificity
from neuron_activation import compute_hidden_layer_activation

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

    # compute_rfs("sae", size_ls=None, num_models=1, num_epochs=60)
    # compute_rfs("dae", size_ls=size_ls, num_models=1, num_epochs=60)

    # analyze_rf_stability("sae", size_ls=None, num_models=1, num_epochs=60)
    # analyze_rf_stability("dae", size_ls=size_ls, num_models=1, num_epochs=60)

    # compute_rf_specificity("sae", num_models=1, size_ls=None, num_epochs=60)
    # compute_rf_specificity("dae", num_models=1, size_ls=size_ls, num_epochs=60)

    compute_hidden_layer_activation("sae", num_models=1, num_epochs=10, epoch=9)
    compute_hidden_layer_activation("dae", num_models=1, num_epochs=60, epoch=9, size_ls=size_ls)

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
