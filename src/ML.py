
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from Preprocessing import prepare_data
from UsefulFunctions import evaluate_classifier
from sklearn import  svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
import numpy as np
import seaborn as sns
from sklearn import metrics
import shap

"""
Script for generating results and evaluating general ML models.
Want to: 
- Load training and testing data.
- Train classifiers on training data.
- Evaluate classifiers on testing data. 
- Generate: confusion matrices, ROC curve, and possibly other plots as well.
"""

classifiers = {
    'Support Vector Machine': svm.SVC(probability=True),
    'K-Nearest Neighbour':KNeighborsClassifier(n_neighbors=10),
    'Random Forest':RandomForestClassifier(n_estimators=10),
    'LightGBM': GradientBoostingClassifier(max_depth =15),
    'XGBoost':xgboost.XGBClassifier(max_depth=15),
}
short_names =  ['SVM','k-NN','RF','LGBM','XGB']

def train_classifiers(classifiers, X_train, y_train):
    for c in classifiers.keys():
        clf = classifiers[c]
        clf.fit(X_train, y_train)
    return classifiers
    
def evaluate_classifiers(classifiers, X_train, y_train, cv):
    for c in classifiers.keys():
        print(c)
        clf = classifiers[c]
        evaluate_classifier(clf, X_train, y_train, cv)

def generate_confusion_matrix(classifiers, name, X_test, y_test):
    classifier = classifiers[name]
    y_hat = classifier.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_hat)
    return confusion_matrix

def plot_confusion_matrices(classifiers, X_test, y_test):
    plt.clf()
    confusion_matrices = {}
    for name in classifiers.keys():
        cm = generate_confusion_matrix(classifiers, name, X_test, y_test) #should be x_test and y_test later
        cm = cm/np.sum(cm) 
        confusion_matrices[name] = cm
    fig, axs = plt.subplots(2,3)
    TP = {}
    TN = {}
    for i, name, ax in zip(range(len(confusion_matrices.keys())), confusion_matrices.keys(), axs.ravel()):
        sns.heatmap(confusion_matrices[name], annot=True, fmt=".1%", cmap='coolwarm', cbar=False, ax=ax,alpha=0.8)
        ax.set_title(name)
        if i == 0 or i == 3:
            ax.set_ylabel('Predicted')
        if i == 3 or i == 4:
            ax.set_xlabel('True')
        ax.set_xticks([])
        ax.set_yticks([])
        TP[name] = confusion_matrices[name][0][0]
        TN[name] = confusion_matrices[name][1][1]
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    axs[-1,-1].bar(x=list(classifiers.keys()), height = list(TP.values()), label='TP', color='red', alpha=0.5)
    axs[-1,-1].bar(x=list(classifiers.keys()), height = list(TN.values()), bottom=list(TP.values()), label = 'TN', color='blue', alpha=0.5) 
    axs[-1,-1].legend()
    axs[-1,-1].set_xticklabels(short_names, rotation=90)
    axs[-1,-1].set_ylabel('Accuracy')
    sns.despine(ax=axs[-1,-1])
    plt.tight_layout()
    plt.savefig('plots/ML/ConfusionMatrices.jpg', dpi=300)

def plot_roc_curve(classifiers, X_test, y_test):
    plt.clf()
    fpr = {}
    tpr = {}
    auc = {}
    colors = plt.cm.coolwarm(np.linspace(0,1,len(classifiers.keys())))
    for name in classifiers.keys():
        y_pred_proba =  classifiers[name].predict_proba(X_test)
        f, t, _ = metrics.roc_curve(y_test,  y_pred_proba[:,1])
        a = metrics.roc_auc_score(y_test, y_pred_proba[:,1])
        fpr[name] = f
        tpr[name] = t
        auc[name] = a
    plt.figure(figsize=(5,3))
    for i, name in enumerate(fpr.keys()):
        plt.plot(fpr[name],tpr[name], label=f"{name} : {auc[name]:.2f}", color=colors[i],alpha=0.5)

    plt.tight_layout()
    plt.legend(title = "AUC",frameon=False)
    plt.ylabel('Sensitivity')
    plt.xlabel('1-specificity')
    sns.despine()

    plt.savefig('plots/ML/ROC.jpg', dpi=300)
        
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
    #evaluate_classifiers(classifiers, X_train, y_train, cv)
    classifiers = train_classifiers(classifiers,X_train, y_train)

    #plot_confusion_matrices(classifiers, X_train, y_train)
    #plot_roc_curve(classifiers, X_train, y_train)
    shap_summary_plot(classifiers, name = 'XGBoost',
                        X = X_train,
                      feature_names = protein_labels)