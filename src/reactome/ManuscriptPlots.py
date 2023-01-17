import pandas as pd
from BINN import BINN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc

def plot_parameters():
    plt.clf()
    covid_ms_hierarchy = "data/reactome/Aaron_covid_HSA_All_ms_path.csv"
    sepsis_ms_hierarchy = "data/reactome/sepsis_HSA_All_ms_path.csv"
    covid_proteins = pd.read_csv('data/ms/covid/AaronQM.tsv', sep="\t")['Protein']
    sepsis_proteins = pd.read_csv('data/ms/sepsis/QuantMatrix.csv',)['Protein']
    trainable_params = {'n':[], 'covid_params':[], 'sepsis_params':[]}
    for n_layers in range(3,7):
        covid_model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=covid_proteins,
                    activation='tanh', 
                    scheduler='plateau',
                    ms_hierarchy = covid_ms_hierarchy)
        sepsis_model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=sepsis_proteins,
                    activation='tanh', 
                    scheduler='plateau',
                    ms_hierarchy = sepsis_ms_hierarchy)
        covid_parameters = covid_model.report_layer_structure()
        c_sparse_parameters = sum(covid_parameters['nz weights'])  + sum(covid_parameters['biases'])
        sepsis_parameters = sepsis_model.report_layer_structure()
        s_sparse_parameters = sum(sepsis_parameters['nz weights'])  + sum(sepsis_parameters['biases'])
        trainable_params['n'].append(n_layers)
        trainable_params['covid_params'].append(c_sparse_parameters)
        trainable_params['sepsis_params'].append(s_sparse_parameters)
    fig, ax = plt.subplots(figsize=(3,3))
    width = 0.5
    x = np.arange(len(trainable_params['n']))
    ax.bar(x=x + width/2, height=trainable_params['covid_params'], width=width, label='COVID', color='blue', alpha=0.5)
    ax.bar(x=x - width/2, height=trainable_params['sepsis_params'], width=width, label='Sepsis', color='red', alpha=0.5)
    for bar in ax.patches:
        if bar.get_height() > 10**6:
            format_string = f"{format(bar.get_height()/10**6, '.1f')}M"
        else:
            format_string = f"{format(bar.get_height()/10**3, '.1f')}k"
        ax.annotate(format_string,
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=7, xytext=(0, 6),
                    textcoords='offset points')
    ax.set_xticks(x, trainable_params['n'])
    plt.xlabel('# layers')
    plt.ylabel('# trainable parameters')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/manuscript/TrainableParameters.svg', dpi=400)
    
    
    
def plot_nodes_per_layer():
    plt.clf()
    covid_ms_hierarchy = "data/reactome/Aaron_covid_HSA_All_ms_path.csv"
    sepsis_ms_hierarchy = "data/reactome/sepsis_HSA_All_ms_path.csv"
    covid_proteins = pd.read_csv('data/ms/covid/AaronQM.tsv', sep="\t")['Protein']
    sepsis_proteins = pd.read_csv('data/ms/sepsis/QuantMatrix.csv',)['Protein']
    nodes = {'n_layers':[], 'covid_nodes':[], 'sepsis_nodes':[], "layer":[]}
    for n_layers in range(3,7):
        covid_model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=covid_proteins,
                    activation='tanh', 
                    scheduler='plateau',
                    ms_hierarchy = covid_ms_hierarchy)
        sepsis_model = BINN(sparse=True,
                    n_layers = n_layers,
                    learning_rate = 0.001, 
                    ms_proteins=sepsis_proteins,
                    activation='tanh', 
                    scheduler='plateau',
                    ms_hierarchy = sepsis_ms_hierarchy)
        
        print(covid_model.column_names[1])
        covid_nr_nodes = [len(x) for x in covid_model.column_names[0:]]
        sepsis_nr_nodes = [len(x) for x in sepsis_model.column_names[0:]]
        
        i = 0
        for covid_nodes, sepsis_nodes in zip(covid_nr_nodes, sepsis_nr_nodes):
            nodes['sepsis_nodes'].append(sepsis_nodes)
            nodes['covid_nodes'].append(covid_nodes)
            nodes['n_layers'].append(n_layers)
            nodes['layer'].append(i+1)
            i = i+1
    nodes = pd.DataFrame(nodes)
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    sns.lineplot(data=nodes, x ='layer', y='covid_nodes', hue='n_layers', palette='vlag',marker='o', ax=ax[0])
    sns.lineplot(data=nodes, x ='layer', y='sepsis_nodes', hue='n_layers', palette='vlag',marker='o', ax=ax[1])
    for a in ax:
        a.set_ylabel('# nodes')
        a.set_xlabel('Hidden layer')
        a.legend(title='# hidden layers', frameon=False)
        a.set_ylim([0,700])
        sns.despine()
    plt.tight_layout()
    plt.savefig('plots/manuscript/NodesPerLayer.svg', dpi=400)
    
    
def plot_copies():
    #harcoded for ease
    
    covid = [0, 15, 65, 172]
    sepsis = [0, 26, 163, 495]
    fig, ax = plt.subplots(figsize=(3,3))
    sns.lineplot(x = [3,4,5,6], y = covid, marker  = 'o', label='COVID', color='blue')
    sns.lineplot(x = [3,4,5,6], y = sepsis, marker  = 'o', label="Sepsis", color='red')
    plt.ylabel('Induced copies in the network')
    plt.xlabel('Number of hidden layers')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/manuscript/Copies.svg', dpi=300)
    
    
def plot_auc():
    plt.clf()
    mean_covid = pd.read_csv(f'plots/manuscript/roc/MeanAUCs_ML_covid.csv', index_col="Unnamed: 0").T
    std_covid = pd.read_csv(f'plots/manuscript/roc/STDAUCs_ML_covid.csv', index_col="Unnamed: 0").T
    mean_sepsis = pd.read_csv(f'plots/manuscript/roc/MeanAUCs_ML_sepsis.csv', index_col="Unnamed: 0").T
    std_sepsis = pd.read_csv(f'plots/manuscript/roc/STDAUCs_ML_sepsis.csv', index_col="Unnamed: 0").T
    
    x_ticks = mean_covid.index
    
    mean_covid = mean_covid.values.flatten()
    std_covid = std_covid.values.flatten()
    mean_sepsis = mean_sepsis.values.flatten()
    std_sepsis = std_sepsis.values.flatten()
    fig, ax = plt.subplots(figsize=(5,3))
    width = 0.45
    x = np.arange(len(x_ticks))
    ax.bar(x=x + width/2, height=mean_covid, yerr =std_sepsis, width=width, label='COVID', color='blue', alpha=0.5)
    ax.bar(x=x - width/2, height=mean_sepsis, yerr=std_sepsis, width=width, label='Sepsis', color='red', alpha=0.5)
    
    for bar in ax.patches:
        format_string = f"{format(bar.get_height(), '.2f')}"
        ax.annotate(format_string,
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=8, xytext=(0, -50),
                    textcoords='offset points')
    short_names =  ['SVM','k-NN','RF','LGBM','XGB']
    
    
    ax.set_xticks(x, short_names)
    ax.set_ylabel('AUC')
    sns.despine()
    plt.legend(frameon=False,loc=(1.04, 0.5))
    plt.tight_layout()
    plt.savefig('plots/manuscript/ML_AUC_BAR.jpg', dpi=300)
    return None


    
if __name__ == '__main__':
    #plot_parameters()
    plot_nodes_per_layer()
    #plot_copies()
    #plot_roc_curve("plots/manuscript/ROCS.jpg")
    #plot_auc()
    #plot_pr_curve("plots/manuscript/PR.jpg")