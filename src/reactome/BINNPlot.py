
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from BINN import BINN
import torch
import numpy as np
from sklearn.metrics import auc


def get_metrics_for_dir(d):
    tot_df = pd.DataFrame()
    d = f'logs/{d}/lightning_logs/'
    for dir in sorted(os.listdir(d)):
        df = pd.read_csv(f'{d}{dir}/metrics.csv')
        df = df.groupby('epoch', as_index=False).mean()
        df['version'] = dir
        tot_df = pd.concat([tot_df, df])
    tot_df.reset_index(inplace=True, drop=True)
    return tot_df


def plot_confusion_matrix(confusion_matrices, save_str):
    plt.clf()
    cm = {'TP':[], 'FP':[], 'FN':[], 'TN':[]}
    for c in confusion_matrices.values():
        p = c[0][0] + c[1][0] 
        n = c[0][1] + c[1][1]
        cm['TP'].append(100*c[0][0]/p)
        cm['FP'].append(100*c[0][1]/n)
        cm['FN'].append(100*c[1][0]/p)
        cm['TN'].append(100*c[1][1]/n)
    cm_mean = [[np.mean(cm['TP']), np.mean(cm['FP'])], [np.mean(cm['FN']), np.mean(cm['TN'])]] # calculate mean for TP, FP, TN, FN
    cm_std = [[np.std(cm['TP']), np.std(cm['FP'])], [np.std(cm['FN']), np.std(cm['TN'])]]
    labels = [ [f'{cm_mean[0][0] : .0f}\u00B1{cm_std[0][0]: .0f}%',f'{cm_mean[0][1]: .0f}\u00B1{cm_std[0][1]: .0f}%' ],
                 [f'{cm_mean[1][0]: .0f}\u00B1{cm_std[1][0]: .0f}%',f'{cm_mean[1][1]: .0f}\u00B1{cm_std[1][1]: .0f}%' ]]
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cm_mean, annot=labels, fmt="", cmap='coolwarm', cbar=False, alpha=0.8)
    plt.tight_layout()
    plt.xticks([0.5,1.5], ['Positive','Negative'])
    plt.yticks([0.5,1.5],  ['Positive','Negative'])
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.savefig(save_str, bbox_inches='tight', dpi=300)
    
def plot_roc_curve(fprs, tprs, aucs, save_str):
    plt.clf()
    tprs = list(tprs.values())
    mean_fpr = list(fprs.values())[0]
    aucs = list(aucs.values())
    mean_tpr = np.mean(tprs, axis=0)
    print(tprs)
    mean_tpr[-1] = 1.0
    std_tpr= np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc= np.std(aucs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.plot(mean_fpr, mean_tpr, label=f"BINN: {mean_auc : .2f} \u00B1 {std_auc : .2f}")
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
    )
    plt.ylabel('Sensitivity')
    plt.xlabel('1-specificity')
    plt.legend(title = "AUC",frameon=False)
    plt.tight_layout()
    sns.despine()
    plt.savefig(save_str, dpi=300)
        

def plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss'):
    plt.clf()
    dirs = [d for d in sorted(os.listdir('logs/')) if d.startswith(test_type)]
    metrics = pd.DataFrame()
    for d in dirs:
        m = get_metrics_for_dir(d)
        m[test_type] = float(d.split('=')[-1])
        metrics = pd.concat([metrics, m])
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(3,3))
    ax = sns.lineplot(data=metrics, x='epoch',y='val_loss', hue=test_type, palette='Reds', ci='sd', err_style='band', alpha=0.8)
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    if 'n_layers' in test_type:
        legend_title = '# layers'
    else:
        legend_title = 'Data split'
    plt.legend(title=legend_title, frameon=False)
    plt.tight_layout()
    plt.ylim([0,1])
    sns.despine()
    plt.savefig(f'plots/BINN/{save_str}.jpg', dpi=300)
    
def plot_val_acc(test_type = 'n_layers', save_str = 'NLayersValLoss'):
    plt.clf()
    dirs = [d for d in sorted(os.listdir('logs/')) if d.startswith(test_type)]
    metrics = pd.DataFrame()
    for d in dirs:
        m = get_metrics_for_dir(d)
        m[test_type] = float(d.split('=')[-1])
        metrics = pd.concat([metrics, m])
        print(f"{d}")
        print(f"Mean accuracy: {(metrics[metrics['epoch'] == 99]['val_acc'].mean())}, {(metrics[metrics['epoch'] == 99]['val_acc'].std())}")
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(3,3))
    
    g = sns.lineplot(data=metrics, x='epoch', y='val_acc', hue=test_type, palette='Blues', ci='sd', alpha=0.8)
    if test_type == 'n_layers':
        legend_title = '# layers'
    else:
        legend_title = 'Data split'
    plt.legend(title=legend_title, frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.tight_layout()
    plt.ylim([0,1.1])
    sns.despine()
    #plt.savefig(f'plots/BINN/{save_str}.jpg', dpi=300)
    
    

def plot_trainable_parameters_over_layers():
    plt.clf()
    ms_proteins = pd.read_csv('data/ms/covid/QuantMatrix.tsv', sep="\t")['Protein']
    trainable_params = {'n':[], 'sparse_params':[], 'dense_params':[]}
    for n_layers in range(3,7):
        model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau')
        parameters = model.report_layer_structure()
        sparse_parameters = sum(parameters['nz weights'])  + sum(parameters['biases'])
        dense_parameters = sum(parameters['weights'])  + sum(parameters['biases'])
        trainable_params['n'].append(n_layers)
        trainable_params['sparse_params'].append(sparse_parameters)
        trainable_params['dense_params'].append(dense_parameters)
    print(trainable_params)
    fig, ax = plt.subplots(figsize=(4,3))
    width = 0.5
    x = np.arange(len(trainable_params['n']))
    ax.bar(x=x + width/2, height=trainable_params['dense_params'], width=width, label='Dense NN', color='blue', alpha=0.5)
    ax.bar(x=x - width/2, height=trainable_params['sparse_params'], width=width, label='BINN', color='red', alpha=0.5)
    for bar in ax.patches:
        if bar.get_height() > 10**6:
            format_string = f"{format(bar.get_height()/10**6, '.0f')}M"
        else:
            format_string = f"{format(bar.get_height()/10**3, '.0f')}k"
        ax.annotate(format_string,
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=10, xytext=(0, -8),
                    textcoords='offset points')
    plt.yscale('log')
    plt.ylim([10**3,1.5*10**7])
    ax.set_xticks(x, trainable_params['n'])
    plt.xlabel('# layers')
    plt.ylabel('# trainable parameters')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/covid/TrainableParameters.jpg', dpi=400)
    
def plot_nodes_per_layer():
    plt.clf()
    ms_proteins = pd.read_csv('data/ms/covid/QuantMatrix.tsv', sep="\t")['Protein']
    nodes = {'n_layers':[], 'number_of_nodes':[], 'layer':[]}
    for n_layers in range(3,7):
        model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau')
        nr_nodes = [len(x) for x in model.column_names[1:]]
        for i, node in enumerate(nr_nodes):
            nodes['number_of_nodes'].append(node)
            nodes['n_layers'].append(n_layers)
            nodes['layer'].append(i+1)
    nodes = pd.DataFrame(nodes)
    print(nodes)
    fig, ax = plt.subplots(figsize=(3,3))
    sns.lineplot(data=nodes, x ='layer', y='number_of_nodes', hue='n_layers', palette='vlag',marker='o')
    plt.ylabel('# nodes')
    plt.xlabel('Hidden layer')
    plt.legend(title='# hidden layers', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/covid/NodesPerLayer.jpg', dpi=400)
    
def plot_copies():
    copies = [0, 0, 72, 367, 1189, 2452] #hardcoded for ease
    fig, ax = plt.subplots(figsize=(3,3))
    sns.lineplot(x = [2,3,4,5,6,7], y = copies, marker  = 'o')
    plt.ylabel('Induced copies in the network')
    plt.xlabel('Number of hidden layers')
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/BINN/Copies.eps', dpi=400)
    

        
if __name__ == '__main__':
    # plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss')
    # plot_val_loss(test_type = 'data_split', save_str = 'DataSplitValLoss')
    #plot_trainable_parameters_over_layers()    
    #plot_performance_of_ensemble('ensemble_voting', 'logs/ensemble_voting/accuracy.csv') # switch this to averaged results and k_means
    #plot_val_acc(test_type = 'n_layers', save_str='NLayersValAcc')
    # plot_val_acc(test_type = 'data_split', save_str = 'DataSplitValAcc')
    
    # plot_val_loss(test_type = 'DENSE_n_layers', save_str = 'DENSENLayersValLoss')
    # plot_val_loss(test_type = 'DENSE_data_split', save_str = 'DENSEDataSplitValLoss')
    # plot_val_acc(test_type = 'DENSE_n_layers', save_str='DENSENLayersValAcc')
    # plot_val_acc(test_type = 'DENSE_data_split', save_str = 'DENSEDataSplitValAcc')
    plot_trainable_parameters_over_layers()
    plot_nodes_per_layer()
    