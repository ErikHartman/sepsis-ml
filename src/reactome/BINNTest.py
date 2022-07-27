
from pytorch_lightning import Trainer
from DataLoaders import MyDataModule, KFoldDataModule, generate_protein_matrix, generate_data, fit_protein_matrix_to_network_input
from BINN import BINN, init_weights
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

## TODO: Look at stochastic weight averaging

def weight_heatmap(layers, file_name, column_names=None, k = 0, only_last = False):
    layer_weights = []
    for l in layers:
        if isinstance(l, nn.Linear):
            df = pd.DataFrame(l.weight.detach().numpy())
            layer_weights.append(df)
  
    for i, layer in enumerate(layer_weights):
        if only_last and i < len(layer_weights)-1: continue
        if column_names:
            layer.columns = column_names[i] 
            if i < len(layer_weights)-1:
                layer.index = column_names[i+1]
        plt.figure(figsize=(20,20))
        sns.heatmap(layer.T, center=0.00, cmap='vlag')
        plt.gca().set_aspect('equal')
        print("Creating weight-heatmap...")
        plt.savefig(f'plots/weight_maps/{file_name}_layer={i}_k={k}.jpg', dpi=200)
        plt.clf()

    
def k_fold(model :  BINN,  k_folds= 4, scale=True, epochs=100):
    columns = model.column_names
    RN_proteins = model.RN.ms_proteins
    model.report_layer_structure()
    protein_matrix = generate_protein_matrix('data/ms')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale =scale)
    batch_size = 32
    for k in range(k_folds):
        model.apply(init_weights) # reset weights. No biases in network
        dataloader = KFoldDataModule(X,y, k=k, num_folds=k_folds, batch_size = batch_size)
        trainer = Trainer(max_epochs=epochs)
        trainer.fit(model, dataloader)
        weight_heatmap(model.layers, 'after_training_test', column_names =columns, k = k, only_last=True)
        trainer.validate(model, dataloader)
        
        
def simple_run(model : BINN, val_size = 0.3, scale=True, epochs=100):
    columns = model.column_names
    RN_proteins = model.RN.ms_proteins
    model.report_layer_structure()
    #weight_heatmap(model.layers, 'before_training', column_names=columns)
    dataloader = MyDataModule(val_size = val_size, RN_proteins = RN_proteins, scale=scale, batch_size=32)
    trainer = Trainer( max_epochs=epochs)
    trainer.fit(model, dataloader)
    weight_heatmap(model.layers, 'after_training_BN', column_names =columns)
    trainer.validate(model, dataloader)
        
if __name__ == '__main__':
    ms_proteins = pd.read_csv('data/ms/proteins.csv')['Proteins']
    model = BINN(sparse=True, learning_rate = 0.001, ms_proteins=ms_proteins, activation='tanh', residual_forward=False)
    #k_fold(model)
    simple_run(model, epochs=100)

    
