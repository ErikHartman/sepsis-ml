
import collections 
import torch.nn as nn
from ReactomeNetwork import ReactomeNetwork
import torch.nn.utils.prune as prune
from pytorch_lightning import LightningModule
import torch
"""
Want a class that creates a neural network (Pytorch) from a ReactomeNetwork.
I figured the main output from the ReactomeNetwork should be the layer connectivity matrices.
These matrices would be multiplied with the dense layers to get a sparse nn.
The sizes of the neural network is also defined by the sizes of the matrices.

THIS IS A MASKING LAYER IN PYTORCH: 
prune.custom_from_mask(
      linLayer, name='weight', mask=torch.tensor(mask))
"""

def generate_sequential(layer_sizes, 
                        connectivity_matrices = None, 
                        activation='tanh'):
    """
    Function that generates a sequential model from layer sizes.
    Need to implement connectivity matrices.
    """
    def append_activation(layers, activation):
        if activation == 'tanh':
            layers.append((f'Tanh {n}', nn.Tanh()))
        elif activation == 'relu':
            layers.append((f'ReLU {n}', nn.ReLU()))
        return layers
        
    layers = []
    for n in range(len(layer_sizes)-1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n+1])
        layers.append((f"Layer {n}", linear_layer))
        if connectivity_matrices is not None:
            # Masking matrix
            prune.custom_from_mask(linear_layer, name='weight', mask=torch.tensor(connectivity_matrices[n].T.values))
        else:
            # Else add dropout layer
            layers.append((f"Droupout {n}", nn.Dropout(0.25)))
            
        append_activation(layers, activation)
    layers.append(("Output layer", nn.Linear(layer_sizes[-1],2))) # Output layer
    model = nn.Sequential(collections.OrderedDict(layers))
    return model



class BINN(LightningModule):
    def __init__(self, sparse=False, n_layers = 4):
        super().__init__()
        RN = ReactomeNetwork()
        connectivity_matrices = RN.get_connectivity_matrices(n_layers)
        layer_sizes = []
        column_names = []
        for matrix in connectivity_matrices:
            i,_ = matrix.shape
            layer_sizes.append(i)
            column_names.append(matrix.index)
        self.layers = generate_sequential(layer_sizes, connectivity_matrices = connectivity_matrices)
        
    def report_layer_structure(self):
        print(self.layers)
        for i, l in enumerate(self.layers):
            if isinstance(l,nn.Linear):
                print(f"Layer {i}")
                print(f"Number of nonzero elements: {torch.count_nonzero(l.weight)}")
                print(f"Total number of elements: {torch.numel(l.weight)}")
        
        
if __name__ == '__main__':
    b = BINN()
    b.report_layer_structure()