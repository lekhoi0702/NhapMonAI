from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


classifier = tree.DecisionTreeClassifier()
classifier.fit(X, y)
plt.figure(figsize=(12, 8))
tree.plot_tree(classifier,feature_names=feature_names,class_names=target_names,filled=True)
plt.show()
