
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from Preprocessing import prepare_data
from sklearn import  svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
import numpy as np
import seaborn as sns
from sklearn import metrics

import shap
from sklearn.model_selection import StratifiedKFold

classifiers = {
    'Support Vector Machine': svm.SVC(probability=True),
    'K-Nearest Neighbour':KNeighborsClassifier(n_neighbors=10),
    'Random Forest':RandomForestClassifier(n_estimators=10),
    'LightGBM': GradientBoostingClassifier(max_depth =15),
    'XGBoost':xgboost.XGBClassifier(max_depth=15),
}
short_names =  ['SVM','k-NN','RF','LGBM','XGB']


def k_fold_confusion_matrices(classifiers, X, y, n_splits = 3):
    plt.clf()
    
    def generate_confusion_matrix(classifier, X_test, y_test):
        y_hat = classifier.predict(X_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_hat)
        return confusion_matrix
    
    confusion_matrices = {}
    cv = StratifiedKFold(n_splits=n_splits)
    
    for name in classifiers.keys():
        classifier = classifiers[name]
        cm = {'TP':[], 'FP':[], 'FN':[], 'TN':[]}
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            c =generate_confusion_matrix(classifier, X[test], y[test])
            p = c[0][0] + c[1][0] 
            n = c[0][1] + c[1][1]
            cm['TP'].append(100*c[0][0]/p)
            cm['FP'].append(100*c[0][1]/n)
            cm['FN'].append(100*c[1][0]/p)
            cm['TN'].append(100*c[1][1]/n)
            print(cm)
        cm_mean = [[np.mean(cm['TP']), np.mean(cm['FP'])], [np.mean(cm['FN']), np.mean(cm['TN'])]] # calculate mean for TP, FP, TN, FN
        cm_std = [[np.std(cm['TP']), np.std(cm['FP'])], [np.std(cm['FN']), np.std(cm['TN'])]]
        confusion_matrices[name] = (cm_mean,cm_std)
    fig, axs = plt.subplots(2,3)
    TP_mean = {}
    TN_mean = {}
    TP_std = {}
    TN_std = {}
    for i, name, ax in zip(range(len(confusion_matrices.keys())), confusion_matrices.keys(), axs.ravel()):
        cm_mean, cm_std = confusion_matrices[name]
        labels = [ [f'{cm_mean[0][0] : .0f}\u00B1{cm_std[0][0]: .0f}%',f'{cm_mean[0][1]: .0f}\u00B1{cm_std[0][1]: .0f}%' ],
                 [f'{cm_mean[1][0]: .0f}\u00B1{cm_std[1][0]: .0f}%',f'{cm_mean[1][1]: .0f}\u00B1{cm_std[1][1]: .0f}%' ]]
        sns.heatmap(cm_mean, annot=labels, fmt="", cmap='coolwarm', cbar=False, ax=ax,alpha=0.8)
        ax.set_title(name)
        if i == 0:
            ax.set_xticks([0.5,1.5], ['Positive','Negative'])
            ax.set_yticks([0.5,1.5],  ['Positive','Negative'])
        else:   
            ax.set_xticks([])
            ax.set_yticks([])
        if i == 0 or i == 3:
            ax.set_ylabel('Predicted')
        if i == 3 or i == 4:
            ax.set_xlabel('True')

        cm_mean, cm_std = confusion_matrices[name]
        TP_mean[name] = cm_mean[0][0] 
        TN_mean[name] = cm_mean[1][1] 
        TP_std[name] = cm_std[0][0] 
        TN_std[name] = cm_std[1][1] 
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)

    axs[-1,-1].bar(x=list(classifiers.keys()), height = list(TP_mean.values()), 
                   yerr = list(TP_std.values()), label='TP', color='red', alpha=0.5)
    axs[-1,-1].bar(x=list(classifiers.keys()), height = list(TN_mean.values()), 
                   bottom=list(TP_mean.values()), yerr = list(TN_std.values()), label = 'TN', color='blue', alpha=0.5) 
    axs[-1,-1].legend()
    axs[-1,-1].set_xticklabels(short_names, rotation=90)
    axs[-1,-1].set_ylabel('True predictions (FP+TP)')
    sns.despine(ax=axs[-1,-1])
    plt.tight_layout()
    plt.savefig('plots/ML/ConfusionMatrices.jpg', dpi=300)

def k_fold_roc(classifiers, X, y, n_splits=3):
    plt.clf()
    cv = StratifiedKFold(n_splits=n_splits)
    colors = plt.cm.coolwarm(np.linspace(0,1,len(classifiers.keys())))
    fig, ax = plt.subplots(figsize=(5,5))
    for k, name in enumerate(classifiers.keys()):
        classifier = classifiers[name]
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            y_pred_proba =  classifiers[name].predict_proba(X[test])
            f, t, _ = metrics.roc_curve(y[test],  y_pred_proba[:,1])
            a = metrics.roc_auc_score(y[test], y_pred_proba[:,1])
            interp_tpr = np.interp(mean_fpr, f, t)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(a)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr,mean_tpr, label=f"{name}: {mean_auc :.2f} \u00B1 {std_auc : .2f}", alpha=1, color=colors[k])
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colors[k],
            alpha=0.2,
        )
            
    plt.legend(title = "AUC",frameon=False)
    plt.ylabel('Sensitivity')
    plt.xlabel('1-specificity')
    plt.tight_layout()
    sns.despine()
    plt.savefig('plots/ML/KFoldROC.jpg', dpi=300)
        
        
def shap_summary_plot(classifiers, name, X, feature_names,):
    plt.clf()
    classifier = classifiers[name]
    if name == "Support Vector Machine": 
        explainer = shap.LinearExplainer(classifier, X)
    elif name == "K-Nearest Neighbour":
        explainer = shap.KernelExplainer(classifier.predict_proba, X)
    else:
        explainer = shap.Explainer(classifier)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, features=X, cmap=plt.cm.coolwarm, feature_names=  feature_names, max_display=8, plot_size=(5,3))
    plt.tight_layout()
    plt.savefig(f'plots/ML/Shap{name}.jpg', dpi=300)

if __name__ == '__main__':
    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    X_train, y_train, protein_labels = prepare_data(scale=True)
    #k_fold_confusion_matrices(classifiers, X_train, y_train)
    #shap_summary_plot(classifiers, name = 'XGBoost', X = X_train,feature_names = protein_labels)
    k_fold_roc(classifiers, X_train, y_train)