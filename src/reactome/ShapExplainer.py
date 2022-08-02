import torch
import shap
import torch.nn as nn
from BINN import BINN
from DataLoaders import generate_protein_matrix, generate_data, fit_protein_matrix_to_network_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""

Good documentation on SHAP: https://christophm.github.io/interpretable-ml-book/shap.html 

"""

if __name__ == '__main__':
    checkpoint_file_path = 'lightning_logs/version_16/checkpoints/epoch=99-step=200.ckpt'
    model = BINN.load_from_checkpoint(checkpoint_file_path)
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

        shap.summary_plot(shap_values[0], test_data, feature_names = feature_names)
        plt.savefig('plots/shap/SHAP_complete_model.jpg')
        plt.clf()

    def shap_for_layers(model, background, test_data):
        feature_index = 0
        intermediate_data = test_data[0:5]
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                feature_names = model.column_names[feature_index]
                explainer = shap.DeepExplainer((model, layer), background[0:20])
                shap_values = explainer.shap_values(test_data[0:5])
    
                shap.summary_plot(shap_values, intermediate_data, feature_names = feature_names)
                plt.savefig(f'plots/shap/SHAP_layer_{feature_index}.jpg')
                feature_index += 1
                plt.clf()
                intermediate_data = layer(intermediate_data)
            if isinstance(layer, nn.Tanh) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                intermediate_data = layer(intermediate_data)

    # shap_for_layers(model, background, test_data)
    shap_test(model, background, test_data)
