from dpks.quant_matrix import QuantMatrix
import pandas as pd
from sklearn import preprocessing
import numpy as np

""" Should include separate methods for preparing training and testing data later on """
def prepare_data(quantification_file =f"data/ms/inner.tsv", design_matrix = f"data/ms/inner_design_matrix.tsv", scale=True, compare=True):
    dm= pd.read_csv(design_matrix, sep='\t')
    GroupOneCols = dm[dm['Group'] == 1]['Sample'].values
    GroupTwoCols = dm[dm['Group'] == 2]['Sample'].values
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