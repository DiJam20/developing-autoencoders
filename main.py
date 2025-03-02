import os


from autoencoder import NonLinearAutoencoder
from solver import *
from model_utils import *
from pc_stability import *

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
    
    angle_matrix("sae", num_models=3)
    angle_matrix("dae", size_ls=size_ls, num_models=3)
    average_angle_matrix_sae, non_computable_cells_sae = average_angle_matrix("sae")
    average_angle_matrix_dae, non_computable_cells_dae = average_angle_matrix("dae")

    heatmap_pc_stability(average_angle_matrix_sae, non_computable_cells_sae, "sae")
    heatmap_pc_stability(average_angle_matrix_dae, non_computable_cells_dae, "dae")

if __name__ == "__main__":
    main()




# PC stability
# RF stability
# Neuron activation given RF (RF specificity)
# Power spectra
# Manipulate values of encodings and see how it affects the decoding
# Manipulate PCs of input images and see which neurons are affected
# Neuron activations in all layers
