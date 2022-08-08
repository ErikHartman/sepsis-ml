from sklearn import datasets
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # figure size
from termcolor import colored as cl # text customization

from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.metrics import accuracy_score # model precision
from sklearn.tree import plot_tree # tree diagram

rcParams['figure.figsize'] = (25, 20)

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = dtc(criterion = 'entropy', max_depth = 3)
model.fit(X_train, y_train)
pred_model = model.predict(X_test)


plt.figure(figsize=(8,8))
plot_tree(model, 
          feature_names=iris.feature_names,  
                   class_names=iris.target_names,
          filled = True, 
          rounded = True)

plt.savefig('plots/background/DecisionTree.jpg', dpi=300)