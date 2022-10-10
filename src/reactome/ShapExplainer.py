import torch
import shap
import torch.nn as nn
from DataLoaders import generate_data, fit_protein_matrix_to_network_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
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

def get_proteins_triv_name(proteins):
    human_proteome = pd.read_csv('data/human_proteome.gz')
    human_proteome['accession'] = human_proteome['accession'].apply(lambda x: x.split('_')[0])
    names = []
    for protein in proteins:
        if protein in human_proteome['accession'].values:
            m = human_proteome.loc[human_proteome['accession'] == protein]['trivname'].values
            assert len(m) == 1
            m = m[0].split('_')[0]
        else:
            m = protein
        names.append(m)
    return names

def get_shorter_names(long_names : list[str]):
    translate = pd.read_csv('ShapShort.csv')
    names = []
    for long_name in long_names:
        long_name = long_name.replace(',', '')
        if long_name in translate['long'].values:
            short_name = translate.loc[translate['long'] == long_name]['short'].values[0].strip("\"")
        else:
            print("Not found: ", long_name)
            short_name = long_name
        names.append(short_name)
    return names

if __name__ == '__main__':
    model_file_path = 'models/full_data_train.pth' # This should be the model that is fully trained (i.e, all data is training data)
    model = torch.load(model_file_path)
    model.report_layer_structure(verbose=True)


    feature_names = model.column_names[0]

    # load data
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale = True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify = y, random_state=42)
    print(len(y_test) + len(y_train))
    print(sum(y_test)/len(y_test), sum(y_train)/len(y_train))
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
                feature_names = get_proteins_triv_name(feature_names)
                explainer = shap.DeepExplainer((model, layer), background)
                shap_values = explainer.shap_values(test_data)
                shap_dict['features'].append(model.column_names[feature_index])
                shap_dict['shap_values'].append(shap_values)
                if plot:
                    shap.summary_plot(shap_values, intermediate_data, feature_names = feature_names, max_display=30, plot_size=[15,6])
                    plt.savefig(f'plots/shap/SHAP_layer_{feature_index}.jpg', dpi=200)
                feature_index += 1
                plt.clf()
                intermediate_data = layer(intermediate_data)
            if isinstance(layer, nn.Tanh) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                intermediate_data = layer(intermediate_data)
        return shap_dict
                

    def shap_sankey(model, background, test_data, show_top_n=10):
        shap_dict = shap_for_layers(model, background, test_data, plot=False)
        feature_dict = {'source':[], 'target':[], 'value':[], 'type':[], 'source layer':[], 'target layer':[]}
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
                        feature_dict['source'].append(f"{features[f]}_{curr_layer}")
                        feature_dict['target'].append(f"{target}_{curr_layer+1}")
                        feature_dict['value'].append(sv_mean[0][f])
                        feature_dict['type'].append(0) 
                        feature_dict['source'].append(f"{features[f]}_{curr_layer}")
                        feature_dict['target'].append(f"{target}_{curr_layer+1}")
                        feature_dict['value'].append(sv_mean[1][f])
                        feature_dict['type'].append(1) 
                        feature_dict['source layer'].append(curr_layer)
                        feature_dict['source layer'].append(curr_layer)
                        feature_dict['target layer'].append(curr_layer+1)
                        feature_dict['target layer'].append(curr_layer+1)
            curr_layer += 1
        n_layers = curr_layer      
        df = pd.DataFrame(data=feature_dict)
        
        def plot_shap_ratio(df, n):
            for layer in df['source layer'].unique():
                shap_ratios = {'sources':[], 'ratio':[], 'mean':[], 'colors':[]}
                plt.clf()
                l = df.loc[df['source layer'] == layer]
                for source in l['source'].unique():
                    s = l.loc[l['source'] == source]
                    val_0 = s.loc[s['type'] == 0]['value'].values[0]
                    val_1 = s.loc[s['type'] == 1]['value'].values[0]
                    ratio = max(val_1/val_0, val_0/val_1)
                    if '_' in source:
                        source = source.split('_')[0]
                    if val_1/val_0 > val_0/val_1:
                        color = 'blue'
                    else:
                        color ='red'
                    shap_ratios['sources'].append(source)
                    shap_ratios['ratio'].append(ratio)
                    shap_ratios['mean'].append(val_0+val_1)
                    shap_ratios['colors'].append(color)

                shap_ratios = pd.DataFrame(shap_ratios)

                shap_ratios.sort_values('mean', ascending=False, inplace=True)
                shap_ratios = shap_ratios[0:n]
                shap_ratios['sources'] = get_proteins_triv_name(shap_ratios['sources'].values.tolist())
                shap_ratios['sources'] = get_pathway_name(shap_ratios['sources'].values.tolist())
                plt.bar(x = shap_ratios['sources'].values, height=shap_ratios['ratio'].values, color=shap_ratios['colors'].values)
                plt.xticks(rotation = 90)
                plt.tight_layout()
                plt.savefig(f'plots/shap/ratio_layer{layer}.jpg',bbox_inches='tight', dpi=300)
                
        #plot_shap_ratio(df, 30)
        
        def save_as_csv(df):
            df_csv = df.copy()
            df_csv['source'] = df_csv['source'].apply(lambda x: x.split('_')[0])
            df_csv['target'] = df_csv['target'].apply(lambda x: x.split('_')[0])
            df_csv['source'] = get_proteins_triv_name(df_csv['source'].values)
            df_csv['source'] = get_pathway_name(df_csv['source'].values)
            df_csv['target'] = get_pathway_name(df_csv['target'].values)
            df_csv.to_csv('ShapConnections.csv')
            
        #save_as_csv(df)
        def normalize_layer_values(df):
            new_df = pd.DataFrame()
            total_value_sum = df['value'].sum()
            for layer in df['source layer'].unique():
                layer_df = df.loc[df['source layer'] == layer]
                layer_total = layer_df['value'].sum()
                layer_df['normalized value'] = total_value_sum *layer_df['value'] / layer_total
                normalized_total = layer_df['normalized value'].sum()
                print(f"Layer total: {layer_total}, normalized total: {normalized_total}")
                new_df = pd.concat([new_df, layer_df])
            return new_df
            
            
        def remove_loops(df):
            """
            If there is a _copy when creating the ReactomeNetwork, there may be loops.
            We should drop rows where source == target 
            """
            df = df[df['source'] != df['target']]
            return df
        
        def remove_double_connections(df):
            """ 
            Since _copy exist, there may be connections from two layers to a single node 
            (one from X and one from X_copy). We only want to plot the value to the earliest layer
            """
            new_df = pd.DataFrame()
            for source in df['source'].unique():
                min_layer = df.loc[df['source'] == source]['source layer'].min() 
                keep = df.loc[df['source'] == source]
                keep = keep.loc[keep['source layer'] == min_layer]
                new_df = pd.concat([new_df, keep])
            return new_df
                 
        #df = remove_loops(df)
        #df = remove_double_connections(df)
        df = normalize_layer_values(df)
        
        def get_top_n(df, layer, n):
            """ Returns the top n (sum of shap) for layer """
            l = df.loc[df['source layer'] == layer]
            s = l.groupby('source', as_index=False).mean().sort_values('value', ascending=False)[0:n]
            top_n_source = s['source'].values.tolist()
            return top_n_source 
        
        top_n = {}
        for layer in range(n_layers):
            if layer < n_layers:
                top_n[layer] = get_top_n(df, layer, show_top_n)

        def set_to_other(row, top_n, source_or_target):
            # Set all that is not in top n to "other"
            s = row[source_or_target]
            if "root_" in s:
                return 'root'
            layer = row['source layer']
            if source_or_target == 'target':
                layer = layer+1
            for t in top_n.values():
                if s in t:
                    return s
            return f"Other connections {layer}"
        
        df['source_w_other'] = df.apply(lambda x: set_to_other(x, top_n, 'source'),axis=1)
        df['target_w_other']= df.apply(lambda x: set_to_other(x, top_n ,'target'),axis=1)

        unique_features = df['source_w_other'].unique().tolist()
        unique_features += df['target_w_other'].unique().tolist()
        code_map, feature_labels = encode_features(list(set(unique_features)))
        sources = df['source_w_other'].values.tolist()

        
        def get_connections(sources, source_target_df):
            conn = source_target_df[source_target_df['source_w_other'].isin(sources)]
            source_code = [get_code(s,code_map) for s in conn['source_w_other']]
            target_code = [get_code(s,code_map) for s in conn['target_w_other']]
            values = [v for v in conn['normalized value']]
            def get_link_color(type,target):
                if 'Other connections' in target:
                    return 'rgb(236,236,236, 0.5)'
                if type == 0:
                    return 'rgba(255,0,0, 0.5)' 
                return 'rgba(0,0,255,0.5)' 
            link_colors = [get_link_color(c,t) for c,t in zip(conn['type'], conn['target_w_other'])]
            return source_code, target_code, values, link_colors
        
        def get_node_colors(sources, df):
            cmaps = {}
            for l in df['source layer'].unique():
                c_df = df[~df['source_w_other'].str.startswith('Other')] # remove Other so that scaling is not messed up
                c_df = c_df[c_df['source layer'] == layer]
                max_value = c_df.groupby('source_w_other').mean()['value'].max()*1.2
                min_value = c_df.groupby('source_w_other').mean()['value'].min()*0.01
                cmap = plt.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin = min_value, vmax=max_value), cmap='Reds')
                cmaps[l] = cmap
            colors = []
            for source in sources:
                if "Other connections" in source:
                    colors.append('rgb(236,236,236, 0.5)')
                elif source == 'root':
                    colors.append('rgba(0,0,0,1)')
                else:
                    source_df = df[df['source_w_other'] == source]
                    intensity = source_df.groupby('source_w_other').mean()['value'].values[0]
                    cmap = cmaps[source_df['source layer'].unique()[0]]
                    r,g,b,a = cmap.to_rgba(intensity, alpha=0.5)
                    colors.append(f"rgba({r*255}, {g*255}, {b*255}, {a})") 
            return colors
        
            
        def get_node_positions(feature_labels, df):
            """ Put residual on bottom and then sort by total outgoing value """
            x = []
            y = []

            grouped_df = df.groupby('source_w_other', as_index=False).agg({'source layer':'min', 'value':'mean'})
            layers = df['source layer'].unique()
            final_df = pd.DataFrame()
            for layer in layers:
                other_df = grouped_df.loc[grouped_df['source_w_other'].str.startswith('Other')]
                other_value = other_df.groupby('source_w_other').mean().value[0]
                
                layer_df = grouped_df[grouped_df['source layer']==layer].sort_values(['value'], ascending=False)
                layer_df = layer_df.loc[~layer_df['source_w_other'].str.startswith('Other')]
                layer_df['rank'] = range(len(layer_df.index))
                layer_df['value'] = layer_df['value'] / layer_df['value'].sum()
                layer_df['y'] = (1-layer_df['value']) / (max(layer_df['rank']))
                layer_df['x'] = (0.01+layer)/(len(layers)+1)
                other_df = pd.DataFrame([[f"Other connections {layer}", layer, other_value, 10, 0.7, (0.01+layer)/(len(layers)+1)]]
                                        ,columns = ['source_w_other','source layer', 'value', 'rank','y','x'])
                final_df = pd.concat([final_df, layer_df, other_df])
            print(final_df)
            for f in feature_labels:
                if f == 'root':
                    x.append(1)
                    y.append(0.5)
                else:
                    
                    x.append(final_df[final_df['source_w_other'] == f]['x'].values[0])
                    y.append(final_df[final_df['source_w_other'] == f]['y'].values[0])
   
            return x,y
        
        
        encoded_source, encoded_target, value, link_colors = get_connections(sources, df)
        node_colors = get_node_colors(feature_labels, df)
        x,y = get_node_positions(feature_labels, df)

        # format text
        i = feature_labels.index('root')
        feature_labels = feature_labels[:i]+['Output']+feature_labels[i+1:]
        feature_labels = [f.split("_")[0] for f in feature_labels] 
        feature_labels = get_pathway_name(feature_labels)
        feature_labels = get_proteins_triv_name(feature_labels)
        feature_labels = get_shorter_names(feature_labels)
        nodes = dict(
                pad = 20,
                thickness = 20,
                line = dict(color = "white", width = 0),
                label = feature_labels,
                color = node_colors,
                x=x,
                y=y
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
                    
                    textfont = dict(size=15,family='Arial'),
                    orientation="h",
                    arrangement = "snap",
                    node = nodes,
                    link = links)
                  ]
            )
        fig.write_image('plots/BINN/ShapSankey.png', width=1750, scale=3, height=1000)
        
  
    #shap_for_layers(model, background, test_data)
    #shap_test(model, background, test_data)
    shap_sankey(model, background, test_data, show_top_n = 10)
