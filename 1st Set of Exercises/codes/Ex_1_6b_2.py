from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt




data = """
x1,x2,x3,x4,y
1,0,1,0,1
0,1,0,1,1
1,0,1,0,1
1,0,1,1,1
0,1,0,0,1
1,0,1,1,-1
0,1,1,0,-1
0,0,0,0,-1
0,0,1,0,-1
1,0,0,0,-1
"""


df = pd.read_csv(StringIO(data))


X = df.drop('y', axis=1)
y = df['y']


clf_gini_max2 = DecisionTreeClassifier(criterion='gini', max_depth=2)
clf_entropy_max2 = DecisionTreeClassifier(criterion='entropy', max_depth=2)


clf_gini_max2.fit(X, y)
clf_entropy_max2.fit(X, y)


y_pred_gini_max2 = clf_gini_max2.predict(X)
y_pred_entropy_max2 = clf_entropy_max2.predict(X)


accuracy_gini_max2 = accuracy_score(y, y_pred_gini_max2)
accuracy_entropy_max2 = accuracy_score(y, y_pred_entropy_max2)

s=['x1','x2','x3','x4','y']

# We will now visualize the trees with a maximum height of 2
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
tree.plot_tree(clf_gini_max2, filled=True, feature_names=s, class_names=['-1', '1'])
plt.title(f'Gini Criterion Decision Tree (Max Depth: 2, Accuracy: {accuracy_gini_max2:.2f})')

plt.subplot(1, 2, 2)
tree.plot_tree(clf_entropy_max2, filled=True, feature_names=s, class_names=['-1', '1'])
plt.title(f'Entropy Criterion Decision Tree (Max Depth: 2, Accuracy: {accuracy_entropy_max2:.2f})')

plt.tight_layout()
plt.show()

(accuracy_gini_max2, accuracy_entropy_max2)
