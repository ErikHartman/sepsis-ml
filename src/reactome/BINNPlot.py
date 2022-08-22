
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
    fig, ax = plt.subplots(figsize=(4,3))
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
    mean_tpr[-1] = 1.0
    std_tpr= np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc= np.std(aucs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fig, ax = plt.subplots(figsize=(4,3))
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
        m['graph'] = m[test_type] + m['version']
        metrics = pd.concat([metrics, m])
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(5,3))
    ax = sns.lineplot(data=metrics, x='epoch',y='val_loss', hue=test_type, palette='rocket', ci='sd', err_style='bars', alpha=0.8)
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    if test_type == 'n_layers':
        legend_title = '# layers'
    else:
        legend_title = 'Data split'
    plt.legend(frameon=False)
    plt.tight_layout()
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
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(5,3))
    g = sns.barplot(data=metrics, x=test_type, y='val_acc', hue=test_type, palette='rocket', ci='sd', alpha=0.8, dodge=False)
    if test_type == 'n_layers':
        plt.xlabel('# layers')
    else:
        plt.xlabel('Data split')
    plt.ylabel('Validation accuracy')
    plt.tight_layout()
    g.legend_.remove()
    sns.despine()
    plt.savefig(f'plots/BINN/{save_str}.jpg', dpi=300)
    
    

def plot_trainable_parameters_over_layers():
    plt.clf()
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    trainable_params = {'n':[], 'sparse_params':[], 'dense_params':[]}
    for n_layers in range(3,7):
        model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau')
        parameters = model.report_layer_structure()
        sparse_parameters = sum(parameters['nz weights']) * 2 # 2 for both weights and biases
        dense_parameters = sum(parameters['weights']) * 2  
        trainable_params['n'].append(n_layers)
        trainable_params['sparse_params'].append(sparse_parameters)
        trainable_params['dense_params'].append(dense_parameters)
    print(trainable_params)
    fig, ax = plt.subplots(figsize=(4,5))
    width = 0.4
    x = np.arange(len(trainable_params['n']))
    ax.bar(x=x + width/2, height=trainable_params['dense_params'], width=width, label='Dense NN', color='blue', alpha=0.5)
    ax.bar(x=x - width/2, height=trainable_params['sparse_params'], width=width, label='Sparse BINN', color='red', alpha=0.5)
    for bar in ax.patches:
        ax.annotate(f"{format(bar.get_height()/10**3, '.0f')}k",
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=10, xytext=(0, 8),
                    textcoords='offset points')
    plt.yscale('log')
    plt.ylim([10**3,1.5*10**7])
    ax.set_xticks(x, trainable_params['n'])
    plt.xlabel('# layers')
    plt.ylabel('# trainable parameters')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/BINN/TrainableParameters.jpg', dpi=400)
    

def plot_performance_of_ensemble(log_dir, ensemble_log_dir):
    k_metrics = get_metrics_for_dir(log_dir)
    ensemble_accuracies = pd.read_csv(ensemble_log_dir)['accuracy'].values
    final_epoch = max(k_metrics['epoch'])
    k_accuracies = k_metrics[k_metrics['epoch'] == final_epoch]['val_acc'].values
   
    fig = plt.figure(figsize=(3,3))
    plt.bar(x=[1,2], height=[np.mean(k_accuracies), np.mean(ensemble_accuracies)], yerr=[np.std(k_accuracies), np.std(ensemble_accuracies)], color=['red','blue'], alpha=0.5, capsize=5)
    plt.ylim([0.5,1.1])
    sns.despine()
    plt.ylabel('Accuracy')
    plt.xticks([1,2], labels=['Individual', 'Ensemble voting'])
    plt.tight_layout()
    plt.savefig('plots/BINN/Accuracies.jpg', dpi=300)
    
def plot_roc(averaged_model):
    """ Plot ROC curve for model. """
    return None
    
def plot_confusion_matrix(confusion_matrix):
    """ Plot confusion for model """
    return None


        
    
if __name__ == '__main__':
    #plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss')
    #plot_val_loss(test_type = 'data_split', save_str = 'DataSplit')
    #plot_trainable_parameters_over_layers()    
    #plot_performance_of_ensemble('ensemble_voting', 'logs/ensemble_voting/accuracy.csv') # switch this to averaged results and k_means
    #plot_val_acc(test_type = 'n_layers', save_str='NLayersValAcc')
    plot_trainable_parameters_over_layers()