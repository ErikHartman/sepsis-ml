
from pytorch_lightning import Trainer
from DataLoaders import MyDataModule
from BINN import BINN
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def weight_heatmap(layers, file_name, column_names=None):
    layer_weights = []
    for l in layers:
        if isinstance(l, nn.Linear):
            df = pd.DataFrame(l.weight.detach().numpy())
            layer_weights.append(df)
    
    for i, layer in enumerate(layer_weights):
        if column_names:
            layer.columns = column_names[i] 
            if i < len(layer_weights)-1:
                layer.index = column_names[i+1]
        plt.figure(figsize=(20,20))
        sns.heatmap(layer.T, center=0.00, cmap='vlag')
        plt.gca().set_aspect('equal')
        plt.savefig(f'plots/weight_maps/{file_name}_{i}.jpg', dpi=200)
        plt.clf()
        
        
if __name__ == '__main__':
    ms_proteins = pd.read_csv('data/ms/proteins.csv')['Proteins']
    model = BINN(sparse=True, learning_rate = 0.01, ms_proteins=ms_proteins, activation='tanh')
    columns = model.column_names
    RN_proteins = model.RN.ms_proteins
    
    model.report_layer_structure()
    weight_heatmap(model.layers, 'before_training', column_names=columns)
    
    dataloader = MyDataModule(val_size = 0.2, RN_proteins = RN_proteins)
    trainer = Trainer(max_epochs=10)


    trainer.fit(model, dataloader)
    weight_heatmap(model.layers, 'after_training', column_names =columns)
    #trainer.validate(model, dataloader)