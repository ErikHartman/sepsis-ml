
from pytorch_lightning import  LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
from dpks.quant_matrix import QuantMatrix
import torch


def generate_protein_matrix(MS_DATA_PATH, scale=False):
    # TODO: Need to merge our proteins with input layer protein list.
    print("Generating protein matrix...")
    design_matrix = pd.read_csv(f'{MS_DATA_PATH}/design_matrix_group_de_early_time_points_subtypes.tsv', sep='\t')

    GroupOneCols = design_matrix[design_matrix['Group'] == 1]['Sample'].values
    GroupTwoCols = design_matrix[design_matrix['Group'] == 2]['Sample'].values

    quant_matrix = QuantMatrix(
        quantification_file=f"{MS_DATA_PATH}/220316_ghost_nrt_filtered.tsv",
        design_matrix_file=f"{MS_DATA_PATH}/design_matrix_group_de_early_time_points_subtypes.tsv"
    )
    df = (
        quant_matrix
        .filter() # filter for q-values (removes rows with low q value (peptides), Q = 0.01) and removes decoys
        .normalize(method="mean", use_rt_sliding_window_filter = True) # best type of normalization is RT-sliding window
        .quantify(method="maxlfq") # play around with minimum_subgroups (default is set 1)
    ).compare_groups(
        method='linregress',
        group_a=1,
        group_b=2,
        min_samples_per_group = 2, 
        level='protein',
    ).to_df()
    
    # TODO: an idea could be to generate the ms_proteins here and return it aswell. Then filtering etc
    # could be done using QuantMatrix and the output will always be consistent with the input to the ReactomeNetwork.
    # DataLoader would then hold ms_proteins in self.
    
    df1 = df[GroupOneCols].T
    df2 = df[GroupTwoCols].T
    y = np.array([1 for x in GroupOneCols] + [2 for x in GroupTwoCols])-1
    X = pd.concat([df1,df2]).fillna(0).to_numpy()
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    return X, y
    
class MyDataModule(LightningDataModule):
    """ Simple LightningDataModule"""
    def __init__(self,  val_size, data_dir = "data/ms",):
        super().__init__()
        self.X, self.y = generate_protein_matrix(data_dir)
        self.val_size = val_size

    def setup(self, stage = None):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=self.val_size)
        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        self.train = torch.utils.data.TensorDataset(X_train, y_train)
        self.val = torch.utils.data.TensorDataset(X_val, y_val)       

    def train_dataloader(self):
        return DataLoader(self.train)

    def val_dataloader(self):
        return DataLoader(self.val)


