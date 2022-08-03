
import collections 
import torch.nn as nn
from ReactomeNetwork import ReactomeNetwork
import torch.nn.utils.prune as prune
import torch
import numpy as np
from numpy import random
from pytorch_lightning import LightningModule


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential( torch.nn.Linear(32, 10),
                                    torch.nn.Linear(10,2))

    def forward(self, x):
        return self.layers(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}
    
    
models = []
for i in range(5):
   models.append(BoringModel())

weights = []
for m in models:
    curr_weights = []
    for l in m.layers:
        if isinstance(l, nn.Linear):
           curr_weights.append(l.weight.detach().numpy())
    weights.append(curr_weights)
weights = np.asarray(weights)
weights = weights.mean(axis=0)
averaged_model = BoringModel()
averaged_model.layers.weight = weights

print(averaged_model.layers.weight[0])