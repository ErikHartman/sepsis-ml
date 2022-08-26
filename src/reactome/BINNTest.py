
from csv import writer
from datetime import datetime
from pytorch_lightning import Trainer
from DataLoaders import MyDataModule, KFoldDataModule, generate_data, fit_protein_matrix_to_network_input
from BINN import BINN,  reset_params
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import auc, confusion_matrix
from BINNPlot import plot_roc_curve, plot_confusion_matrix
from torchmetrics import ROC
from sklearn.model_selection import train_test_split
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

    
def k_fold(model :  BINN,  k_folds = 4, scale=True, epochs=100, log_name = '', save=False, save_prefix='', data_split=1.0, X=None, y=None, plot=False):
    trained_models = []
    RN_proteins = model.RN.ms_proteins
    if X is None:
        protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
        protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins)
        X,y = generate_data(protein_matrix, 'data/ms', scale =scale)
    if data_split < 1:
        X, _, y,_ = train_test_split(X,y, train_size = data_split, stratify = y)
        print('Number of data-points: ', len(y))
    fprs = {}
    tprs = {}
    aucs = {}
    confusion_matrices = {}
    for k in range(k_folds):
        log_dir = f'logs/{log_name}'
        logger = SuperLogger(log_dir, tensorboard = True, csv =  True)
        model.apply(reset_params) # parameters in network
        dataloader = KFoldDataModule(X,y, k=k, num_folds=k_folds, batch_size = 32)
        trainer = Trainer(logger=logger.get_logger_list(), max_epochs=epochs)
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader, ckpt_path='best')
        if plot:
            mean_fpr = np.linspace(0, 1, 100)
            y_true = []
            X_val = []
            val_data = dataloader.data_val # TensorDataset
            for i in range(len(val_data)): # iterate over dataset
                X_i, y_i = val_data[i]
                y_true.append(y_i.numpy().tolist())
                X_val.append(X_i.numpy().tolist())
            X_val  = torch.Tensor(X_val)
            y_true = torch.LongTensor(y_true)
            roc = ROC( pos_label=1)
            with torch.no_grad():
                pred = model(X_val)
            pred = torch.argmax(pred, dim=1)
            fpr, tpr, thresholds = roc(pred, y_true)
            interp_tpr = np.interp(mean_fpr, fpr.numpy(), tpr.numpy())
            interp_tpr[0] = 0.0
  
            fprs[k] = mean_fpr
            tprs[k] = interp_tpr
            aucs[k] = auc(fpr,tpr)
            confusion_matrices[k] = confusion_matrix(y_true, pred)
        if save:
            torch.save(model, f'models/{save_prefix}_k={k}.pth') 
        trained_models.append(copy.deepcopy(model))
    if plot:
        plot_roc_curve(fprs, tprs, aucs, f'plots/BINN/ROC{save_prefix}.jpg')
        plot_confusion_matrix(confusion_matrices,  f'plots/BINN/ConfusionMatrix{save_prefix}.jpg')
    return trained_models
        
def simple_run(model : BINN, val_size = 0.3, scale=True, epochs=100, log_name = '', callbacks = [], fit=True, validate=True, dataloader=None):
    """
    Performs a simple run for the model.
    """
    log_dir = f'logs/{log_name}'
    logger = SuperLogger(log_dir, tensorboard = True, csv =  True)
    RN_proteins = model.RN.ms_proteins
    model.report_layer_structure()
    if dataloader is None:
        dataloader = MyDataModule(val_size = val_size, RN_proteins = RN_proteins, scale=scale, batch_size=32, protein_matrix_path = 'data/ms/QuantMatrix.csv')
    trainer = Trainer(callbacks = callbacks, logger = logger.get_logger_list(), max_epochs=epochs)
    
    if fit:
        trainer.fit(model, dataloader)
    if validate:
        trainer.validate(model, dataloader)  
    return model
    
def ensemble_voting(trained_models : list, X_test : torch.Tensor, y_test : torch.LongTensor):
    y_hats = []
    for model in trained_models:
        y_hat = model(X_test)
        y_hat = torch.argmax(y_hat, dim=1).numpy().tolist()
        y_hats.append(y_hat)
    
    ensemble_vote = []
    y_hats_sum = np.sum(np.array(y_hats), axis=0)
    y_hats_sum = y_hats_sum/len(y_hats)
    for i in y_hats_sum:
        if i > 0.5:
            ensemble_vote.append(1)
        if i < 0.5:
            ensemble_vote.append(0)
    y_test = y_test.numpy()
    accuracy = sum(y_test == ensemble_vote) / len(y_test)
    print(f'Ensemble accuracy {100*accuracy :.2f}%')
    with open('logs/ensemble_voting/accuracy.csv', 'a') as file:
        w = writer(file)
        w.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"),accuracy])
        file.close()
    return accuracy
 
def ensemble_learning(model :BINN, save_prefix='ensemble_voting', save=False, epochs = 100):
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins=model.RN.ms_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale =True)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
    trained_models = k_fold(model, log_name=save_prefix, k_folds=3, scale=True, epochs=epochs, save=save, save_prefix=save_prefix,  X=X_train, y=y_train)
    ensemble_voting(trained_models, torch.Tensor(X_test), torch.LongTensor(y_test))

def k_fold_with_varying_n_layers(n_layers_list=  [3,4,5,6], save=False, sparse=True):
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    for n in n_layers_list:
        model = BINN(sparse=sparse,
                    n_layers = n,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau')
        if sparse is True:
            save_prefix = f'n_layers={n}'
        else:
            save_prefix = f"DENSE_n_layers={n}"
        trained_models = k_fold(model, log_name=save_prefix, k_folds=3, scale=True, epochs=100, save=save, save_prefix=save_prefix)
    return trained_models
        
def k_fold_with_varying_data(model :BINN, data_splits=[0.25,0.5,0.75,1], save=False, n_layers = 4, sparse=True):
    for data_split in data_splits:
        if sparse is True:
            save_prefix = f'data_split={data_split}'
        else:
            save_prefix = f'DENSE_data_split={data_split}'
        trained_models = k_fold(model, log_name=save_prefix, k_folds=3, scale=True, epochs=100, save=save, save_prefix=save_prefix, data_split=data_split)
    return trained_models
        
def train_on_full_data(model : BINN, save_prefix="full_data_train", save=False, epochs=100):
    log_dir = f'logs/{save_prefix}'
    logger = SuperLogger(log_dir, tensorboard = True, csv =  True)
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale =True)
    dataloader= torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)), 
                                            batch_size = 8,
                                            num_workers=12, 
                                            shuffle=True)
    trainer = Trainer(callbacks = [], logger = logger.get_logger_list(), max_epochs=epochs)
    trainer.fit(model, dataloader)
    if save:
        torch.save(model, f'models/{save_prefix}.pth')
    
    
if __name__ == '__main__':
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    model = BINN(sparse=True,
                 n_layers = 4,
                 learning_rate = 0.001, 
                 ms_proteins=ms_proteins,
                 activation='tanh', 
                 scheduler='plateau', 
                 validate=False)
    #k_fold_with_varying_n_layers(save=True, sparse=False)
    #k_fold(model, k_folds=3, epochs = 100, plot=False, save=False)
    train_on_full_data(model, save=True)
    
    #weight_heatmap(model.layers, 'after_training_averaged_model', column_names =model.column_names, only_last=True)
    
    
    #k_fold_with_varying_data(model, save=True, sparse=False)

   
    #
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=15, verbose=False, mode="min")
    # callbacks = [early_stop_callback]
    #model = simple_run(model, epochs=100)
    # model.weights
    #torch.save(model, 'models/test.pth')

    
