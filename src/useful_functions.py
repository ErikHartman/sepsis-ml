import math
from scipy import stats
import numpy as np

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