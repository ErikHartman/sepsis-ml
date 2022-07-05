import itertools
import networkx as nx
import numpy as np
import pandas as pd
from ReactomeNetwork import ReactomeNetwork


reactome_net = ReactomeNetwork()
print(reactome_net.info())
print('# of root nodes {} , # of terminal nodes {}'.format(len(reactome_net.get_roots()),
                                                           len(reactome_net.get_terminals())))

print("Completed Tree " , nx.info(reactome_net.get_completed_tree(n_levels=5)))
print("Completed network ", nx.info(reactome_net.get_completed_network(n_levels=5)))
layers = reactome_net.get_layers(n_levels=10)
print(len(layers))

def get_map_from_layer(layer_dict):
    '''
    :param layer_dict: dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}
    :return: dataframe map of layer (index = proteins, columns = pathways, , values = 1 if connected; 0 else)
    '''
    pathways = layer_dict.keys()
    proteins = list(itertools.chain.from_iterable(layer_dict.values()))
    proteins = list(np.unique(proteins))
    df = pd.DataFrame(index=pathways, columns=proteins)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T

print('Layer shapes')
for i, layer in enumerate(layers[::-1]):
    mapp = get_map_from_layer(layer)
    if i == 0:
        proteins = list(mapp.index)
    filter_df = pd.DataFrame(index=proteins)
    all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
    proteins = list(mapp.columns)
    print(f"Layer {i}")
    print(all.shape)

"""
_all_ spits out dataframes for each layer. These dataframes have a 1 if connected, else 0.

These are the masking matrices.

Could we now create a dense NN with the correct shape (from layers_shape) and just multiply 
with the corresponding masking layer after each layer?

This would create the sparse NN we're after.

"""
    
