import pandas as pd
from DataLoaders import *
from ReactomeNetwork import ReactomeNetwork

ms_proteins = pd.read_csv('data/ms/proteins.csv')['Proteins']
RN = ReactomeNetwork(filter=True, ms_proteins=ms_proteins)
conn_mat = RN.get_connectivity_matrices(n_levels=5)[0]
proteins = conn_mat.index.values
protein_matrix = generate_protein_matrix()
protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, proteins)

