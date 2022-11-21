
from csv import writer
from datetime import datetime
from pytorch_lightning import Trainer
from DataLoaders import MyDataModule, KFoldDataModule, generate_data, fit_protein_matrix_to_network_input
from BINN import BINN,  reset_params
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import auc, confusion_matrix
from BINNPlot import plot_roc_curve, plot_confusion_matrix
from torchmetrics import ROC, PrecisionRecallCurve
from sklearn.model_selection import train_test_split
from Loggers import SuperLogger
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
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

    
def k_fold(model :  BINN,  k_folds = 5, scale=True, epochs=100, log_name = 'k_fold', save=False, save_prefix='', data_split=1.0, X=None, y=None, plot=False,
           protein_matrix = 'data/ms/covid/QuantMatrix.tsv', design_matrix = 'data/ms/covid/design_cropped.csv', dataset="covid", impute=True):
    trained_models = []
    RN_proteins = model.RN.ms_proteins
    if X is None:
        if protein_matrix.endswith('.tsv'):
            sep = "\t"
        else:
            sep = ","
        protein_matrix = pd.read_csv(protein_matrix, sep=sep)
        protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins)
        X,y = generate_data(protein_matrix, design_matrix = design_matrix, scale = scale, group_column = 'group'
                            , sample_column = 'sample', group_one=1, group_two = 2, impute=impute)
    if data_split < 1:
        X, _, y,_ = train_test_split(X,y, train_size = data_split, stratify = y)
        print('Number of data-points: ', len(y))
    fprs = {}
    tprs = {}
    aucs = {}
    prs = {}
    pr_auc = {}
    confusion_matrices = {}
    for k in range(k_folds):
        log_dir = f'logs/{log_name}'
        logger = SuperLogger(log_dir, tensorboard = True, csv =  True)
        model.apply(reset_params) # parameters in network
        dataloader = KFoldDataModule(X,y, k=k, num_folds=k_folds, batch_size = 8)
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.05, patience=15, stopping_threshold = 0.9, verbose=False, mode="max")
        trainer = Trainer(logger=logger.get_logger_list(), max_epochs=epochs, callbacks=[])
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader, ckpt_path='best')
        if plot:
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)
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
            pr_curve = PrecisionRecallCurve(pos_label=1)
            with torch.no_grad():
                pred = model(X_val)
            probabilities = F.softmax(pred, dim=1)[:, 1]
            confusion_matrix_pred = torch.argmax(pred, dim=1)
            f, t, _ = roc(probabilities, y_true)
            p, r, _ = pr_curve(probabilities, y_true)
            interp_tpr = np.interp(mean_fpr, f.numpy(),t.numpy())
            interp_tpr[0] = 0.0
            fprs[k] = mean_fpr
            tprs[k] = interp_tpr
            aucs[k] = auc(f,t)

            prs[k] = np.interp(mean_recall, p, r)
            pr_auc[k] = auc(r,p)
            confusion_matrices[k] = confusion_matrix(y_true, confusion_matrix_pred)
        if save:
            torch.save(model, f'models/{save_prefix}_k={k}.pth') 
        trained_models.append(copy.deepcopy(model))
    if plot:
        pd.DataFrame(fprs, index=range(len(fprs[0]))).to_csv(f'plots/manuscript/roc/fprs_{dataset}.csv')
        pd.DataFrame(tprs, index=range(len(tprs[0]))).to_csv(f'plots/manuscript/roc/tprs_{dataset}.csv')
        pd.DataFrame(aucs, index=[0]).to_csv(f'plots/manuscript/roc/aucs_{dataset}.csv')
        pd.DataFrame(prs, index=range(len(prs[0]))).to_csv(f'plots/manuscript/precision_recall/prs_{dataset}.csv')
        pd.DataFrame(pr_auc, index=[0]).to_csv(f'plots/manuscript/precision_recall/aucs_{dataset}.csv')
        plot_roc_curve(fprs, tprs, aucs, f'plots/manuscript/{dataset}_ROC{save_prefix}.jpg')
        plot_confusion_matrix(confusion_matrices,  f'plots/manuscript/{dataset}_ConfusionMatrix{save_prefix}.jpg')
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


def k_fold_with_varying_n_layers(n_layers_list=  [3,4,5,6], save=False, sparse=True, weight = 100/torch.Tensor([74,123])):
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    for n in n_layers_list:
        model = BINN(sparse=sparse,
                    n_layers = n,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau',
                    weight = weight)
        if sparse is True:
            save_prefix = f'n_layers={n}'
        else:
            save_prefix = f"DENSE_n_layers={n}"
        trained_models = k_fold(model, log_name=save_prefix, k_folds=5, scale=True, epochs=100, save=save, save_prefix=save_prefix)
    return trained_models
        
def k_fold_with_varying_data(model :BINN, data_splits=[0.2, 0.4, 0.6, 0.8 ,1], save=False, sparse=True):
    for data_split in data_splits:
        if sparse is True:
            save_prefix = f'data_split={data_split}'
        else:
            save_prefix = f'DENSE_data_split={data_split}'
        trained_models = k_fold(model, log_name=save_prefix, k_folds=5, scale=True, epochs=100, save=save, save_prefix=save_prefix, data_split=data_split)
    return trained_models
        
def train_on_full_data(model : BINN, protein_matrix, design_matrix, save_prefix="full_data_train", save=False, epochs=100, impute=True):
    log_dir = f'logs/{save_prefix}'
    logger = SuperLogger(log_dir, tensorboard = True, csv =  True)
    if protein_matrix.endswith('csv'):
        sep = "\,"
    else:
        sep = "\t"
    protein_matrix = pd.read_csv(protein_matrix, sep = sep)
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
    #X,y = generate_data(protein_matrix, 'data/ms', scale =True)
    X,y = generate_data(protein_matrix, design_matrix = design_matrix, scale = True, group_column = 'group'
                            , sample_column = 'sample', group_one=1, group_two = 2, impute=True)
    dataloader= torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)), 
                                            batch_size = 8,
                                            num_workers=12, 
                                            shuffle=True)
    trainer = Trainer(callbacks = [], logger = logger.get_logger_list(), max_epochs=epochs)
    trainer.fit(model, dataloader)
    if save:
        torch.save(model, f'models/manuscript/{save_prefix}.pth')
    
    
if __name__ == '__main__':
    impute = False
    #dataset = "sepsis"
    #dataset = "covid" 
    dataset = "aaron"
    covid_ms_hierarchy = "data/reactome/covid_HSA_All_ms_path.csv"
    sepsis_ms_hierarchy = "data/reactome/sepsis_HSA_All_ms_path.csv"
    aaron_covid_ms_hierarchy = "data/reactome/Aaron_covid_HSA_All_ms_path.csv"
    if dataset == "covid":
        ms_hierarchy = covid_ms_hierarchy
        weight = 100 / torch.Tensor([281,406]) # class weights
        ms_proteins = pd.read_csv('data/ms/covid/QuantMatrix.tsv', sep="\t")['Protein']
        protein_matrix = 'data/ms/covid/QuantMatrix.tsv'
        design_matrix = 'data/ms/covid/design_cropped.tsv'
        k_folds = 6
        n_layers = 5
        epochs = 50
    elif dataset == 'aaron':
        ms_hierarchy = aaron_covid_ms_hierarchy
        weight = 100 / torch.Tensor([281,406]) # class weights
        ms_proteins = pd.read_csv('data/ms/covid/AaronQM.tsv', sep="\t")['Protein']
        protein_matrix = 'data/ms/covid/AaronQM.tsv'
        design_matrix = 'data/ms/covid/design_cropped.tsv'
        k_folds = 6
        n_layers = 4
        epochs = 30
        
    elif dataset == "sepsis":
        ms_proteins = pd.read_csv('data/ms/sepsis/QuantMatrixNoNA.csv', sep=",")['Protein']
        ms_hierarchy = sepsis_ms_hierarchy
        weight = 100 / torch.Tensor([74,123]) # class weights
        protein_matrix = 'data/ms/sepsis/QuantMatrixNoNA.csv'
        design_matrix = 'data/ms/sepsis/inner_design_matrix.tsv'
        k_folds = 3
        n_layers = 4
        epochs = 50
        
    
    model = BINN(sparse=True,
                 n_layers = n_layers,
                 learning_rate = 0.001, 
                 ms_proteins = ms_proteins,
                 activation = 'tanh', 
                 scheduler = 'plateau', 
                 validate = False, 
                 weight = weight,
                 ms_hierarchy = ms_hierarchy
                 )
    model.report_layer_structure(verbose=True)
    #k_fold(model, k_folds=k_folds, epochs = epochs, protein_matrix = protein_matrix, design_matrix = design_matrix,
     #      plot=True, save=False, log_name=f"{dataset}_k_fold", dataset = dataset, save_prefix=f"{dataset}_NLayers_{n_layers}")

    train_on_full_data(model,protein_matrix, design_matrix, save=True, epochs=epochs, save_prefix=f"{dataset}_full_data_train_{epochs}_epochs", impute=impute)
    
    #weight_heatmap(model.layers, 'after_training_averaged_model', column_names =model.column_names, only_last=True)
    
    
    #k_fold_with_varying_data(model, save=True, sparse=False)
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=15, verbose=False, mode="min")
    # callbacks = [early_stop_callback]
    #model = simple_run(model, epochs=100)
    # model.weights
    #torch.save(model, 'models/test.pth')

    
