from dpks.quant_matrix import QuantMatrix
import pandas as pd
from sklearn import preprocessing
import numpy as np

""" Should include separate methods for preparing training and testing data later on """
def prepare_data(quantification_file =f"data/ms/220316_ghost_nrt_filtered.tsv", design_matrix = f"data/ms/design_matrix_group_de_early_time_points_subtypes.tsv", scale=True):
    design_matrix = pd.read_csv(f'data/ms/design_matrix_group_de_early_time_points_subtypes.tsv', sep='\t')
    GroupOneCols = design_matrix[design_matrix['Group'] == 1]['Sample'].values
    GroupTwoCols = design_matrix[design_matrix['Group'] == 2]['Sample'].values
    quant_matrix = QuantMatrix(
        quantification_file=quantification_file,
        design_matrix_file=f"data/ms/design_matrix_group_de_early_time_points_subtypes.tsv")
    df = (
        quant_matrix
        .filter()
        .normalize(method="mean", use_rt_sliding_window_filter = True)
        .quantify(method="maxlfq") 
    ).compare_groups(
        method='linregress',
        group_a=1,
        group_b=2,
        min_samples_per_group = 2, # play around with this 
        level='protein',
    ).to_df().dropna(subset=['CorrectedPValue'])
    protein_labels = df['Protein'].values
    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    df1.columns = protein_labels
    df2.columns = protein_labels
    y = np.array([1 for x in GroupOneCols] + [2 for x in GroupTwoCols]) - 1 

    df_X = pd.concat([df1,df2]).fillna(0)
    X = df_X.to_numpy()
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    return X, y, protein_labels