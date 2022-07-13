
import collections 
import torch.nn as nn
from ReactomeNetwork import ReactomeNetwork
import torch.nn.utils.prune as prune
from pytorch_lightning import Trainer, LightningModule, seed_everything
from DataLoaders import MyDataModule
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
    def __init__(self, ms_proteins = [], learning_rate = 1e-4, sparse=False, n_layers = 4):
        super().__init__()
        self.RN = ReactomeNetwork(ms_proteins = ms_proteins, filter=True)
        connectivity_matrices = self.RN.get_connectivity_matrices(n_layers)
        layer_sizes = []
        self.column_names = []
       
        for matrix in connectivity_matrices:
            i,_ = matrix.shape
            layer_sizes.append(i)
            self.column_names.append(matrix.index)
            
        if sparse:
            self.layers = generate_sequential(layer_sizes, connectivity_matrices = connectivity_matrices)
        else:
            self.layers = generate_sequential(layer_sizes)    
        self.init_weights(self.layers)   
        self.loss = nn.CrossEntropyLoss() 
        self.learning_rate = learning_rate 
    
    def forward(self, x):
        return self.layers(x) 
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log('val_loss',loss)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
    def report_layer_structure(self):
        print(self.layers)
        for i, l in enumerate(self.layers):
            if isinstance(l,nn.Linear):
                print(f"Layer {i}")
                print(f"Number of nonzero elements: {torch.count_nonzero(l.weight)}")
                print(f"Total number of elements: {torch.numel(l.weight)}")
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def init_weights(self,m):
        # This still works when pruning
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
        
    def calculate_accuracy(self, y, prediction):
        return torch.sum(y == prediction).item() / (float(len(y)))
        
        
if __name__ == '__main__':
    model = BINN()
    ms_proteins = model.RN.get_ms_proteins()
    print(ms_proteins)
    print(model.layers)
    model.report_layer_structure()
    dataloader = MyDataModule(0.2)
    seed_everything(42, workers=True)
    trainer = Trainer(deterministic=True, max_epochs=50,)
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)