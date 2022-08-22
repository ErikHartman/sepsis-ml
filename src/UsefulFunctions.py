import math
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def color_if_significant(row):
    if row['logPValue'] > 1.31 and row['Log2FoldChange1-2'] < -1.2:
        return 'blue'
    if row['logPValue'] > 1.31 and row['Log2FoldChange1-2'] > 1.2:
        return 'red'
    return 'black'

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


def evaluate_classifier(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv)
    f1_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
    recall = cross_val_score(clf, X, y, cv=cv, scoring='recall')
    precision = cross_val_score(clf, X, y, cv=cv, scoring='precision')

    print("Accuracy: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("F1: %0.2f with a standard deviation of %0.2f" % (f1_scores.mean(), f1_scores.std()))
    print("Recall: %0.2f accuracy with a standard deviation of %0.2f" % (recall.mean(), recall.std()))
    print("Precision: %0.2f with a standard deviation of %0.2f" % (precision.mean(), precision.std()))
    
    
def save_protein_list(protein_list, save_path):
    protein_df = pd.DataFrame(protein_list, columns=['Proteins'])
    protein_df.to_csv(save_path, index=False)