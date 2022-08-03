
import collections 
import torch.nn as nn
from ReactomeNetwork import ReactomeNetwork
import torch.nn.utils.prune as prune
from pytorch_lightning import LightningModule
import torch


def generate_sequential(layer_sizes, 
                        connectivity_matrices = None, 
                        activation='tanh'):
    """
    Generates a sequential model from layer sizes.
    """
    def append_activation(layers, activation):
        if activation == 'tanh':
            layers.append((f'Tanh {n}', nn.Tanh()))
        elif activation == 'relu':
            layers.append((f'ReLU {n}', nn.ReLU()))
        elif activation == "leaky relu":
            layers.append((f'LeakyReLU {n}', nn.LeakyReLU()))
        return layers
        
    layers = []
    for n in range(len(layer_sizes)-1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n+1], bias=False)
        layers.append((f"Layer_{n}", linear_layer)) # linear layer 
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n+1]))) # batch normalization
        if connectivity_matrices is not None:
            # Masking matrix
            prune.custom_from_mask(linear_layer, name='weight', mask=torch.tensor(connectivity_matrices[n].T.values))
            layers.append((f"Dropout_{n}", nn.Dropout(0.1)))
        else:
            # If not pruning do dropout instead.
            layers.append((f"Dropout_{n}", nn.Dropout(0.5)))
        append_activation(layers, activation)
    layers.append(("Output layer", nn.Linear(layer_sizes[-1],2, bias=False))) # Output layer
    model = nn.Sequential(collections.OrderedDict(layers))
    return model


class BINN(LightningModule):
    def __init__(self, 
                 ms_proteins : list = [], 
                 activation : str = 'tanh', 
                 learning_rate : float = 1e-4, 
                 sparse : bool = False, 
                 n_layers : int = 4, 
                 scheduler : str = 'plateau'):
        super().__init__()
        if sparse:
            self.RN = ReactomeNetwork(ms_proteins = ms_proteins, filter=True)
        else:
            self.RN = ReactomeNetwork()
        print(self.RN.info())
        connectivity_matrices = self.RN.get_connectivity_matrices(n_layers)
        layer_sizes = []
        self.column_names = []
        for matrix in connectivity_matrices:
            i,_ = matrix.shape
            layer_sizes.append(i)
            self.column_names.append(matrix.index)
        if sparse:
            self.layers = generate_sequential(layer_sizes, connectivity_matrices = connectivity_matrices, activation=activation)
        else:
            self.layers = generate_sequential(layer_sizes, activation=activation)    
        init_weights(self.layers)   
        self.loss = nn.CrossEntropyLoss() 
        self.learning_rate = learning_rate 
        self.scheduler = scheduler
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.layers(x) 
        
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
    def report_layer_structure(self):
        print(self.layers)
        for i, l in enumerate(self.layers):
            if isinstance(l,nn.Linear):
                print(f"Layer {i}")
                print(f"Number of nonzero elements: {torch.count_nonzero(l.weight)}")
                print(f"Total number of elements: {torch.numel(l.weight)}")
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        if self.scheduler == 'plateau':
            scheduler = {"scheduler": 
                        torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, patience=10, 
                            threshold = 0.00001, 
                            mode='min', verbose=True),
                        "interval": "epoch",
                        "monitor": "val_loss"}
        elif self.scheduler == 'step':
            scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, verbose=True)}
        return [optimizer], [scheduler]
    
    def calculate_accuracy(self, y, prediction):
        return torch.sum(y == prediction).item() / (float(len(y)))
        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        
def reset_weights(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()