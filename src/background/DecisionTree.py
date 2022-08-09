from sklearn import datasets
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # figure size
from termcolor import colored as cl # text customization

from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm
from sklearn.model_selection import train_test_split # splitting the data
from matplotlib.colors import ListedColormap, to_rgb
from sklearn.tree import plot_tree # tree diagram
import numpy as np

rcParams['figure.figsize'] = (25, 20)

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = dtc(criterion = 'entropy', max_depth = 3)
model.fit(X_train, y_train)
pred_model = model.predict(X_test)


plt.figure(figsize=(8,8))
artists = plot_tree(model, 
          feature_names=iris.feature_names,  
                   class_names=iris.target_names,
          filled = True, 
          rounded = True)
colors = ['#3b4cc0','#dddddd','#b40426']
for artist, impurity, value in zip(artists, model.tree_.impurity, model.tree_.value):
    # let the max value decide the color; whiten the color depending on impurity (gini)
    r, g, b = to_rgb(colors[np.argmax(value)])

    artist.get_bbox_patch().set_facecolor((r, g, b,0.8))
    artist.get_bbox_patch().set_edgecolor('black')

plt.savefig('plots/background/DecisionTree.jpg', dpi=300)