from dpks.quant_matrix import QuantMatrix
import pandas as pd
from sklearn import preprocessing
import numpy as np

""" Should include separate methods for preparing training and testing data later on """
def prepare_data(quantification_file =f"data/ms/sepsis/inner.tsv", 
                 design_matrix = f"data/ms/sepsis/inner_design_matrix.tsv", 
                 group_column = "Group",
                 scale=True, 
                 compare=True):
    dm = pd.read_csv(design_matrix, sep='\t')
    GroupOneCols = dm[dm[group_column] == 1]['Sample'].values
    GroupTwoCols = dm[dm[group_column] == 2]['Sample'].values
    quant_matrix = QuantMatrix(
        quantification_file=quantification_file,
        design_matrix_file=design_matrix)
    qm = (
        quant_matrix
        .normalize(method="mean", use_rt_sliding_window_filter=True) # best type of normalization is RT-sliding window
        .quantify(method="maxlfq") # play around with minimum_subgroups (default is set 1)
    )
    
    if compare:
        df = qm.compare_groups(
            method='linregress',
            group_a=1,
            group_b=2,
            min_samples_per_group = 2, 
            level='protein',
        ).to_df().dropna(subset=['CorrectedPValue'])
        print(f"{len(df.index)} proteins kept after dropping NaN p-value")
    else:
        df = qm.to_df()
        print(f"{len(df.index)} proteins")
    protein_labels = df['Protein'].values
    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    df1.columns = protein_labels
    df2.columns = protein_labels
    y = np.array([1 for x in GroupOneCols] + [2 for x in GroupTwoCols]) - 1 
    print('Concatenating')
    df = pd.concat([df1,df2]).fillna(0)
    X = df.to_numpy()
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    return X, y, protein_labels

def prepare_HA_data(quantification_file =f"data/ms/HA/QuantMatrix.csv", design_matrix = f"data/ms/HA/design.csv", group_column = 'RandomizationCode'):
    df = pd.read_csv(quantification_file)
    design = pd.read_csv(design_matrix)
    GroupOneCols = design[design[group_column] == 0]['SampleName']
    GroupTwoCols = design[design[group_column] == 1]['SampleName']
    protein_labels = df['Protein'].values

    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    df1.columns = protein_labels
    df2.columns = protein_labels
    y = np.array([0 for x in GroupOneCols] + [1 for x in GroupTwoCols])
    df_X = pd.concat([df1,df2]).fillna(0)
    X = df_X.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, protein_labels

def prepare_sepsis_data(quantification_file =f"data/ms/sepsis/QuantMatrix.csv", design_matrix = f"data/ms/sepsis/inner_design_matrix.tsv",
                        group_column = 'group'):
    df = pd.read_csv(quantification_file)
    design = pd.read_csv(design_matrix, sep = "\t")
    GroupOneCols = design[design[group_column] == 1]['sample']
    GroupTwoCols = design[design[group_column] == 2]['sample']
    protein_labels = df['Protein'].values

    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    df1.columns = protein_labels
    df2.columns = protein_labels
    y = np.array([0 for x in GroupOneCols] + [1 for x in GroupTwoCols])
    df_X = pd.concat([df1,df2]).fillna(0)
    X = df_X.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, protein_labels

def prepare_covid_data(quantification_file =f"data/ms/covid/AaronQM.tsv", design_matrix = f"data/ms/covid/design_cropped.tsv", group_column = 'group'):
    df = pd.read_csv(quantification_file, sep="\t")
    design = pd.read_csv(design_matrix, sep = "\t")
    GroupOneCols = design[design[group_column] == 1]['sample']
    GroupTwoCols = design[design[group_column] == 2]['sample']
    protein_labels = df['Protein'].values

    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    df1.columns = protein_labels
    df2.columns = protein_labels
    y = np.array([0 for x in GroupOneCols] + [1 for x in GroupTwoCols])
    df_X = pd.concat([df1,df2]).fillna(0)
    X = df_X.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, y, protein_labels