from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

# For the purpose of this example, we will create a simple dataset
# Let's consider a case with 4 features (x1, x2, x3, x4)
# Each feature can be either 0 (false) or 1 (true)

X = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]])

Y = (X[:, 0] | ~X[:, 1]) | (X[:, 2] | X[:, 3])

# Create and train the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Plotting the tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=['x1', 'x2', 'x3', 'x4'], class_names=['False', 'True'])
plt.show()
