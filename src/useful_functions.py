import math
from scipy import stats
import numpy as np
import pandas as pd

def remove_nan(array):
    return [x for x in array if not(math.isnan(x)) == True]

def get_p_value(row, GroupOneCols, GroupTwoCols):
    groupOneValues = remove_nan(row[GroupOneCols].values)
    groupTwoValues = remove_nan(row[GroupTwoCols].values)
    statistic, pvalue = stats.ttest_ind(groupOneValues, groupTwoValues)
    return pvalue

def get_fold_change(row, GroupOneCols, GroupTwoCols):
    groupOneValues = remove_nan(row[GroupOneCols].values)
    groupTwoValues = remove_nan(row[GroupTwoCols].values)
    log_fold_change = np.mean(groupOneValues) / np.mean(groupTwoValues)
    return log_fold_change

def get_difference(row, GroupOneCols, GroupTwoCols):
    val_one = remove_nan(row[GroupOneCols].values)
    val_two = remove_nan(row[GroupTwoCols].values)
    
    return np.mean(val_one)- np.mean(val_two)

def add_protein_trivnames(my_df, protein_col_name):
    protein_df = pd.read_csv('data/human_proteome.gz')
    protein_df = protein_df[['accession','trivname']]
    my_df = my_df.merge(protein_df, left_on=protein_col_name, right_on='accession', how='left')
    return my_df

