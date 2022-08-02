import itertools
import networkx as nx
import numpy as np
import pandas as pd
from ReactomeNetwork import ReactomeNetwork

ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
reactome_net1 = ReactomeNetwork(filter=True, ms_proteins=ms_proteins)
reactome_net2 = ReactomeNetwork()
for reactome_net in [reactome_net1, reactome_net2]:
    print(reactome_net.info())
    # print('# of root nodes {} , # of terminal nodes {}'.format(len(reactome_net.get_roots()),
    #                                                            len(reactome_net.get_terminals())))

    # print("Completed Tree " , nx.info(reactome_net.get_completed_tree(n_levels=5)))
    # print("Completed network ", nx.info(reactome_net.get_completed_network(n_levels=5)))
    # layers = reactome_net.get_layers(n_levels=5)
    connectivity_matrices = reactome_net.get_connectivity_matrices(n_levels=5)
    for matrix in connectivity_matrices:
        print(matrix.shape)
        i,j = matrix.shape
        print(f"Number of connections in matrix: {matrix.values.sum()}")
