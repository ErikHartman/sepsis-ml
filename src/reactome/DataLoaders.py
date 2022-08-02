
from pytorch_lightning import  LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dpks.quant_matrix import QuantMatrix
import torch


def generate_protein_matrix(MS_DATA_PATH = "data/ms", save=False):
    print("Generating protein matrix...")
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
    if save:
        df.to_csv('data/ms/QuantMatrix.csv')
    return df

def fit_protein_matrix_to_network_input(protein_matrix, RN_proteins):
    # df needs to be sorted and filtered on RN_proteins.
    nr_proteins_in_matrix = len(protein_matrix.index)
    if len(RN_proteins) > nr_proteins_in_matrix:
        # if we have more proteins in our network than in ms data (only occurs if sparse = False in BINN)
        # Then we want to merge on RN_proteins 
        RN_df = pd.DataFrame(RN_proteins, columns=['Protein'])
        protein_matrix = protein_matrix.merge(RN_df, how='right', on='Protein')
        print(protein_matrix)
    if len(RN_proteins) > 0:
        # This sorts the protein matrix on the input nodes from the BINN.
        protein_matrix.set_index('Protein', inplace=True)
        protein_matrix = protein_matrix.loc[RN_proteins]
    return protein_matrix

def generate_data(protein_matrix, MS_DATA_PATH, scale=False):
    design_matrix = pd.read_csv(f'{MS_DATA_PATH}/design_matrix_group_de_early_time_points_subtypes.tsv', sep='\t')
    GroupOneCols = design_matrix[design_matrix['Group'] == 1]['Sample'].values
    GroupTwoCols = design_matrix[design_matrix['Group'] == 2]['Sample'].values
    df1 = protein_matrix[GroupOneCols].T
    df2 = protein_matrix[GroupTwoCols].T
    y = np.array([1 for x in GroupOneCols] + [2 for x in GroupTwoCols])-1
    X = pd.concat([df1,df2]).fillna(0).to_numpy()
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    return X, y

    
class MyDataModule(LightningDataModule):
    """ Simple LightningDataModule"""
    def __init__(self,  
                 val_size : int = 0.3, 
                 data_dir : str  = "data/ms",
                 RN_proteins : list = [], 
                 scale : bool = False, 
                 batch_size : int = 8,
                 num_workers: int = 12,
                 protein_matrix_path = None):
        super().__init__()
        if protein_matrix_path is not None:
            protein_matrix = pd.read_csv(protein_matrix_path)
        else:
            protein_matrix = generate_protein_matrix(data_dir)
        protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins)
        self.X, self.y = generate_data(protein_matrix, data_dir, scale)
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=self.val_size)
        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        self.train = torch.utils.data.TensorDataset(X_train, y_train)
        self.val = torch.utils.data.TensorDataset(X_val, y_val)       

    def train_dataloader(self):
        return DataLoader(self.train, num_workers = self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, num_workers = self.num_workers, batch_size= self.batch_size)



class KFoldDataModule(LightningDataModule):
    def __init__(
            self,
            X, 
            y,
            k: int = 1,  # fold number
            split_seed: int = 42,  # split needs to be always the same for correct cross validation
            num_folds: int = 10,
            num_workers: int = 12,
            batch_size: int = 8,
        ):
        super().__init__()
        self.X = X
        self.y = y
        self.k = k
        self.split_seed = split_seed
        self.num_folds = num_folds
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None
        self.batch_size = batch_size

    def setup(self, stage = None):
        if not self.data_train and not self.data_val:
            # choose fold to train on
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.split_seed)
            all_splits = [k for k in kf.split(self.X, self.y)]
            print("Fold: ", self.k)
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.astype(int), val_indexes.astype(int)
            X_train = self.X[train_indexes]
            X_val = self.X[val_indexes]
            y_train = [self.y[i] for i in train_indexes]
            y_val = [self.y[i] for i in val_indexes]
            print('Fraction class 1 in y_val: ', frac_i(y_val, 1))
            print('Validation indexes:' , val_indexes)
            self.data_train = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
            self.data_val = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.data_train, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.data_val, batch_size = self.batch_size, num_workers=self.num_workers)

def frac_i(l, i):
    nr_el = len(l)
    sum_i=  0
    for el in l:
        if el == i:
            sum_i += 1
    return sum_i/nr_el