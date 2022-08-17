
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from BINN import BINN
import torch
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
        

def plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss', save_individual = False):
    plt.clf()
    dirs = [d for d in sorted(os.listdir('logs/')) if d.startswith(test_type)]
    metrics = pd.DataFrame()
    for d in dirs:
        m = get_metrics_for_dir(d)
        m[test_type] = d
        m['graph'] = m[test_type] + m['version']
        metrics = pd.concat([metrics, m])
    metrics.reset_index(inplace=True, drop=True)
    plt.figure(figsize=(5,3))
    print(metrics)
    ax = sns.lineplot(data=metrics, x='epoch',y='val_loss', hue=test_type, palette='rocket', ci='sd', err_style='bars', alpha=0.5)
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.legend(frameon=False)
    plt.tight_layout()
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
    plt.figure(figsize=(3,3))   
    df = pd.DataFrame.from_dict(data=trainable_params)
    print(trainable_params)
    fig, ax = plt.subplots()
    width = 0.4
    x = np.arange(len(trainable_params['n']))
    ax.bar(x=x + width/2, height=trainable_params['dense_params'], width=width, label='Dense BINN', color='blue', alpha=0.5)
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
    plt.legend()
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


""" Accuracy of averaged models with n_layers """
def plot_performance_of_n_layers(models=[]):
    for m in models:
        model = torch.load(m)
        
    
if __name__ == '__main__':
    #plot_val_loss(test_type = 'n_layers', save_str = 'NLayersValLoss')
    #plot_val_loss(test_type = 'data_split', save_str = 'DataSplit')
    plot_trainable_parameters_over_layers()    
    #plot_performance_of_averaged_model('n_layers=5', 'n_layers=6') # switch this to averaged results and k_means