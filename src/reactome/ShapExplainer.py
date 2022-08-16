import torch
import shap
import torch.nn as nn
from DataLoaders import generate_data, fit_protein_matrix_to_network_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def get_pathway_name(features):
    mapping = pd.read_csv('data/reactome/ReactomePathways.txt', sep='\t', names=['ReactomeID','Name','Species'])
    mapping = mapping[mapping['Species'] == 'Homo sapiens']
    names= []
    for feature in features:
        if feature in mapping['ReactomeID'].values:
            m = mapping[mapping['ReactomeID'] == feature]['Name'].values
            assert len(m) == 1
            m = m[0]
        else:
            m = feature
        
        names.append(m)
    return names
    

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
        feature_dict = {'source':[], 'target':[], 'value':[], 'type':[], 'source layer':[]}
        connectivity_matrices = model.get_connectivity_matrices()
        curr_layer = 0
        
        def encode_features(features):
            feature_map = {'feature':features, 'code':list(range(len(features)))}
            feature_map = pd.DataFrame(data=feature_map)
            return feature_map, features
        
        def get_code(feature, feature_map):
            code = feature_map[feature_map['feature'] == feature]['code'].values[0]
            return code

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
                    feature_dict['type'].append(0) 
                    feature_dict['source'].append(features[f])
                    feature_dict['target'].append(target)
                    feature_dict['value'].append(sv_mean[0][f])
                    feature_dict['type'].append(1) 
                    feature_dict['source layer'].append(curr_layer)
                    feature_dict['source layer'].append(curr_layer)
            curr_layer += 1
        n_layers = curr_layer      
        df = pd.DataFrame(data=feature_dict)
        
        def normalize_layer_values(df):
            """ TODO: double check this """
            new_df = pd.DataFrame()
            total_value_sum = df['value'].sum()
            for layer in df['source layer'].unique():
                layer_df = df[df['source layer'] == layer]
                layer_total = layer_df['value'].sum()
                print(layer_total)
                layer_df['normalized value'] = total_value_sum *layer_df['value'] / layer_total
                new_df = pd.concat([new_df, layer_df])
            return new_df
            
            
        def remove_loops(df):
            """
            If there is a _copy when creating the ReactomeNetwork, there may be loops.
            We should drop rows where source == target 
            """
            df = df[df['source'] != df['target']]
            return df
            
            
        """ Want to turn this source-target dataframe into a sankey diagram """
        df = remove_loops(df)
        df = normalize_layer_values(df)
        df['source'] = get_pathway_name(df['source'].values)
        df['target'] = get_pathway_name(df['target'].values)
        
        def get_top_n(df, layer, n):
            """ Returns the top n (sum of shap) for layer """
            l = df[df['source layer'] == layer]
            l = l.groupby('source', as_index=False).sum().sort_values('value', ascending=False)[0:n]
            return l['source'].values.tolist()
        
        top_n = {}
        for layer in range(n_layers):
            if layer < n_layers:
                top_n[layer] = get_top_n(df, layer, 10)

        
        # Set all that is not in top n to "other"
        def set_to_other(row, top_n, source_or_target):
            s = row[source_or_target]
            if s == 'root':
                return 'root'
            layer = row['source layer']
            if source_or_target == 'target':
                layer = layer+1
            for t in top_n.values():
                if s in t:
                    return s
            return f"Residual {layer}"
        
        df['target_w_other']= df.apply(lambda x: set_to_other(x, top_n ,'target'),axis=1)
        df['source_w_other'] = df.apply(lambda x: set_to_other(x, top_n, 'source'),axis=1)
        print(df)

        unique_features = df['source_w_other'].unique().tolist()
        unique_features += df['target_w_other'].unique().tolist()
        code_map, feature_labels = encode_features(unique_features)
        sources = df['source_w_other'].values.tolist()
        
        def get_connections(sources, source_target_df):
            conn = source_target_df[source_target_df['source_w_other'].isin(sources)]
            source_code = [get_code(s,code_map) for s in conn['source_w_other']]
            target_code = [get_code(s,code_map) for s in conn['target_w_other']]
            values = [v for v in conn['normalized value']]
            def get_link_color(type,target):
                if 'Residual' in target:
                    return 'rgb(236,236,236)'
                if type == 0:
                    return 'rgba(255,0,0, 0.5)' 
                return 'rgba(0,0,255,0.5)' 
            link_colors = [get_link_color(c,t) for c,t in zip(conn['type'], conn['target_w_other'])]
            return source_code, target_code, values, link_colors
        
        def get_node_colors(sources, df):
            colors = []
            for source in sources:
                if "Residual" in source:
                    colors.append('rgb(236,236,236)')
                elif source == 'root':
                    colors.append('rgba(0,0,0,1)')
                else:
                    source_df = df[df['source_w_other'] == source]
                    red = source_df[source_df['type'] == 0].groupby('source_w_other').sum()['value'].values[0]
                    blue = source_df[source_df['type'] == 1].groupby('source_w_other').sum()['value'].values[0]
                    red = 255*red/(red+blue)
                    blue = 255*blue/(red+blue)
                    colors.append(f'rgba({red},0,{blue},0.5)')
            return colors
        
        def get_node_positions(feature_labels, df):
            """ Put residual on bottom and then sort by total outgoing value """
            x = []
            y = []
            grouped_df = df.groupby('source_w_other', as_index=False).agg({'source layer':'min', 'value':'sum'})
            layers = df['source layer'].unique()
            final_df = pd.DataFrame()
            for layer in layers:
                layer_df = grouped_df[grouped_df['source layer']==layer].sort_values(['value'], ascending=False)
                layer_df['rank'] = range(len(layer_df.index))
                layer_df['y'] = (1+layer_df['rank']) / (max(layer_df['rank'])+1)
                layer_df['x'] = layer/len(layers)
                final_df = pd.concat([final_df, layer_df])
            for f in feature_labels:
                if f == 'root':
                    x.append(1)
                    y.append(0.5)
                else:
                    x.append(final_df[final_df['source_w_other'] == f]['x'].values[0])
                    y.append(final_df[final_df['source_w_other'] == f]['y'].values[0])
            return x,y
        
        encoded_source, encoded_target, value, link_colors = get_connections(sources, df)
        node_colors = get_node_colors(feature_labels,df)

        x,y = get_node_positions(feature_labels,df) #not used because tis messes up
        
        nodes = dict(
                pad = 25,
                thickness = 20,
                line = dict(color = "black", width = 0),
                label = feature_labels,
                color = node_colors,
                )
        links = dict(
                source =encoded_source,
                target = encoded_target,
                value = value,
                color = link_colors
                )
        
        fig = go.Figure(
            data=[
                go.Sankey(
                    textfont = dict(size=15),
                    orientation="h",
                    arrangement = "snap",
                    domain = dict(x=[0,1], y=[0,1]),
                    node = nodes,
                    link = links)
                  ]
            )

        
        fig.write_image('plots/BINN/ShapSankey.png', width=1500, height=1000, scale=3)
        
  
    #shap_for_layers(model, background, test_data)
    #shap_test(model, background, test_data)
    shap_sankey(model, background, test_data)
