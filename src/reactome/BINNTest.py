
from pytorch_lightning import Trainer
from DataLoaders import MyDataModule, KFoldDataModule, generate_data, fit_protein_matrix_to_network_input
from BINN import BINN,  reset_weights
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Loggers import SuperLogger
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import copy
import numpy as np


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

    
def k_fold(model :  BINN,  k_folds = 4, scale=True, epochs=100):
    trained_models = []
    RN_proteins = model.RN.ms_proteins
    model.report_layer_structure()
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale =scale)
    for k in range(k_folds):
        model.apply(reset_weights) # reset weights. No biases in network
        dataloader = KFoldDataModule(X,y, k=k, num_folds=k_folds, batch_size = 32)
        trainer = Trainer( max_epochs=epochs)
        trainer.fit(model, dataloader)
        weight_heatmap(model.layers, 'after_training_ensemble_learning', column_names =model.column_names, k = k, only_last=True)
        trainer.validate(model, dataloader)
        trained_models.append(copy.deepcopy(model))
    return trained_models
        
def simple_run(model : BINN, val_size = 0.3, scale=True, epochs=100, log_name = '', callbacks = [], fit=True, validate=True):
    logger = SuperLogger(log_name, tensorboard = True, csv =  True)
    RN_proteins = model.RN.ms_proteins
    model.report_layer_structure()
    #weight_heatmap(model.layers, 'before_training', column_names= model.column_names)
    dataloader = MyDataModule(val_size = val_size, RN_proteins = RN_proteins, scale=scale, batch_size=32, protein_matrix_path = 'data/ms/QuantMatrix.csv')
    trainer = Trainer(callbacks = callbacks, logger = logger.get_logger_list(), max_epochs=epochs)
    
    if fit:
        trainer.fit(model, dataloader)
    #weight_heatmap(model.layers, 'after_training_BN', column_names = model.column_names)
    if validate:
        trainer.validate(model, dataloader)
    return model
    
    
def ensemble_learning(k_folds = 3, epochs=100, n_layers = 4, save=False):
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    model = BINN(sparse=True,
                 n_layers = n_layers,
                 learning_rate = 0.001, 
                 ms_proteins=ms_proteins,
                 activation='tanh', 
                 scheduler='plateau')
    trained_models = k_fold(model, k_folds=k_folds, scale=True, epochs=epochs)
    averaged_model = BINN(sparse=True,
                 n_layers = n_layers,
                 learning_rate = 0.001, 
                 ms_proteins=ms_proteins,
                 activation='tanh', 
                 scheduler='plateau')
    weights = []
    for m in trained_models:
        curr_weights = []
        for l in m.layers:
            if isinstance(l, nn.Linear):
                curr_weights.append(l.weight.detach().numpy())
        weights.append(curr_weights)
    weights = np.asarray(weights)
    weights = weights.mean(axis=0)
    averaged_model = model
    averaged_model.layers.weight = weights
    simple_run(averaged_model, fit=False, validate=True)
    if save:
        torch.save(averaged_model, 'models/averaged_model.pth')
    return averaged_model
    
        
if __name__ == '__main__':
    
    model = ensemble_learning(save=True, epochs=50)
    
    weight_heatmap(model.layers, 'after_training_averaged_model', column_names =model.column_names, only_last=True)
    # ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    # model = BINN(sparse=True,
    #              n_layers = 4,
    #              learning_rate = 0.001, 
    #              ms_proteins=ms_proteins,
    #              activation='tanh', 
    #              scheduler='plateau')
    # #k_fold(model, k_folds=3)
    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=15, verbose=False, mode="min")
    # callbacks = [early_stop_callback]
    # model = simple_run(model, callbacks = callbacks, epochs=10)
    # model.weights
    #torch.save(model, 'models/test.pth')

    
