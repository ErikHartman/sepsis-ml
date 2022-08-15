import torch
import shap
import torch.nn as nn
from DataLoaders import generate_data, fit_protein_matrix_to_network_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""

Good documentation on SHAP: https://christophm.github.io/interpretable-ml-book/shap.html 

"""

def get_pathway_name(features):
    mapping = pd.read_csv('data/reactome/ReactomePathways.txt', sep='\t', names=['ReactomeID','Name','Species'])
    mapping = mapping[mapping['Species'] == 'Homo sapiens']
    mapping = mapping[mapping['ReactomeID'].isin(features)]
    mapping.set_index('ReactomeID', inplace=True)
    try:
        mapping = mapping.loc[features]
    except KeyError:
        return features
    return mapping['Name'].values
    

if __name__ == '__main__':
    model_file_path = 'models/test.pth'
    model = torch.load(model_file_path)
    model.report_layer_structure()


    feature_names = model.column_names[0]

    # load data
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale = True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
    X_test = torch.Tensor(X_test)
    y_test = torch.LongTensor(y_test)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    background = X_train
    test_data = X_test


    def shap_test(model, background, test_data):
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_data)

        shap.summary_plot(shap_values, test_data, feature_names = feature_names)
        plt.savefig('plots/shap/SHAP_complete_model.jpg')
        plt.clf()

    def shap_for_layers(model, background, test_data, plot=True):
        feature_index = 0
        intermediate_data = test_data
        shap_dict = {'features':[], 'shap_values':[]}
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                feature_names = model.column_names[feature_index]
                feature_names = get_pathway_name(feature_names)
                explainer = shap.DeepExplainer((model, layer), background)
                shap_values = explainer.shap_values(test_data)
                shap_dict['features'].append(model.column_names[feature_index])
                shap_dict['shap_values'].append(shap_values)
                if plot:
                    shap.summary_plot(shap_values, intermediate_data, feature_names = feature_names, plot_size=[12,6])
                    plt.savefig(f'plots/shap/SHAP_layer_{feature_index}.jpg', dpi=200)
                feature_index += 1
                plt.clf()
                intermediate_data = layer(intermediate_data)
            if isinstance(layer, nn.Tanh) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                intermediate_data = layer(intermediate_data)
        return shap_dict
                

    def shap_sankey(model, background, test_data):
        """ 
        Want a methods that produces Sankey diagram for neural network
        from a to b: value = shap_value
        
        """
        shap_dict = shap_for_layers(model, background, test_data, plot=False)
        feature_dict = {'source':[], 'target':[], 'value':[], 'type':[]}
        connectivity_matrices = model.get_connectivity_matrices()
        for sv, features, cm in zip(shap_dict['shap_values'],shap_dict['features'], connectivity_matrices):
            # first dim: positive vs negative class, second dim: for each test data, third dim: for each feature
            sv = np.asarray(sv)
            sv = abs(sv)
            sv_mean = np.mean(sv, axis=1) #mean(|shap_value|) = impact on model class  
                     
            for f in range(sv_mean.shape[-1]):
                connections = cm[cm.index == features[f]]
                connections = connections.loc[:, (connections != 0).any(axis=0)] # get targets and append to target
               
                for target in connections:
                    feature_dict['source'].append(features[f])
                    feature_dict['target'].append(target)
                    feature_dict['value'].append(sv_mean[1][f])
                    feature_dict['type'].append('0') 
                    feature_dict['source'].append(features[f])
                    feature_dict['target'].append(target)
                    feature_dict['value'].append(sv_mean[0][f])
                    feature_dict['type'].append('1') 
        df = pd.DataFrame(data=feature_dict)
        """ Want to turn this source-target dataframe into a sankey diagram """
        print(df)
        print(df[df['source'] == 'R-HSA-1643685'])
                
                
  
    #shap_for_layers(model, background, test_data)
    #shap_test(model, background, test_data)
    shap_sankey(model, background, test_data)
