
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from BINN import BINN
import numpy as np


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
        

def plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss'):
    plt.clf()
    dirs = [d for d in sorted(os.listdir('logs/')) if d.startswith(test_type)]
    metrics = pd.DataFrame()
    for d in dirs:
        m = get_metrics_for_dir(d)
        m['# layers'] = d
        metrics = pd.concat([metrics, m])
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(5,3))
    ax = sns.lineplot(data=metrics, x='epoch',y='val_loss',hue='# layers', palette='rocket')
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.legend(frameon=False)
    plt.tight_layout()
    sns.despine()
    plt.savefig(f'plots/BINN/{save_str}.jpg', dpi=300)
    

def plot_trainable_parameters_over_layers():
    plt.clf()
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    trainable_params = {'n':[], 'params':[]}
    for n_layers in range(3,7):
        model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=ms_proteins,
                    activation='tanh', 
                    scheduler='plateau')
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        trainable_params['n'].append(n_layers)
        trainable_params['params'].append(params)
    plt.figure(figsize=(3,3))   
    df = pd.DataFrame.from_dict(data=trainable_params)
    sns.barplot(data=df, x = 'n', y= 'params', color='red', alpha=0.5)
    plt.xlabel('# layers')
    plt.ylabel('# Trainable parameters')

    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/BINN/TrainableParameters.jpg', dpi=300)
    

def plot_performance_of_averaged_model(log_dir, averaged_log_dir):
    k_metrics = get_metrics_for_dir(log_dir)
    averaged_metrics = get_metrics_for_dir(averaged_log_dir)
    final_epoch = max(k_metrics['epoch'])
    k_accuracies = k_metrics[k_metrics['epoch'] == final_epoch]['val_acc'].values
    averaged_accuracy = averaged_metrics[averaged_metrics['epoch'] == final_epoch]['val_acc'].values[0]
    fig = plt.figure(figsize=(3,3))
    plt.bar(x=[1,2], height=[np.mean(k_accuracies), averaged_accuracy], yerr=[np.std(k_accuracies), 0], color=['red','blue'], alpha=0.5, capsize=5)
    plt.ylim([0.5,1])
    sns.despine()
    plt.ylabel('Accuracy')
    plt.xticks([1,2], labels=['Individual', 'Averaged'])
    plt.tight_layout()
    plt.savefig('plots/BINN/Accuracies.jpg', dpi=300)
    
    
if __name__ == '__main__':
    plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss')
    plot_val_loss(test_type = 'data_split', save_str = 'DataSplit')
    plot_trainable_parameters_over_layers()    
    plot_performance_of_averaged_model('n_layers=5', 'n_layers=6') # switch this to averaged results and k_means