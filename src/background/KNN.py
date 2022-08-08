import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X = iris.data
y = iris.target


pca = PCA(n_components = 2)
X2 = pca.fit_transform(X)
clf = neighbors.KNeighborsClassifier(n_neighbors=10
                                     )
clf.fit(X2, y)# Plotting decision region
plt.figure(figsize=(5,5))
plot_decision_regions(X2, y, clf=clf, colors='#3b4cc0,#dddddd,#b40426', legend=2)

plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.tight_layout()
plt.savefig('plots/background/KNN.jpg', dpi=300)