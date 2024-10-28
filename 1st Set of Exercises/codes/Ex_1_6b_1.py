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


clf_gini = DecisionTreeClassifier(criterion='gini')
clf_entropy = DecisionTreeClassifier(criterion='entropy')


X = df.drop('y', axis=1)
y = df['y']


clf_gini.fit(X, y)
clf_entropy.fit(X, y)


y_pred_gini = clf_gini.predict(X)
y_pred_entropy = clf_entropy.predict(X)


accuracy_gini = accuracy_score(y, y_pred_gini)
accuracy_entropy = accuracy_score(y, y_pred_entropy)


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)

s=['x1', 'x2', 'x3', 'x4']

#'''
tree.plot_tree(clf_gini, filled=True, feature_names=s, class_names=['-1', '1'])
plt.title(f'Decision Tree using Gini (Accuracy: {accuracy_gini:.2f})')

plt.subplot(1, 2, 2)
tree.plot_tree(clf_entropy, filled=True, feature_names=s, class_names=['-1', '1'])
plt.title(f'Decision Tree using Entropy (Accuracy: {accuracy_entropy:.2f})')

plt.tight_layout()
plt.show()

(accuracy_gini, accuracy_entropy)
#'''