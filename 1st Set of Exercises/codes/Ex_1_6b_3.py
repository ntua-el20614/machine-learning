from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
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

max_depth_limit = 5


def find_optimal_tree_depth(X, y, max_depth_limit, criterion='gini'):
    best_score = 0
    best_depth = 1
    
    for depth in range(1, max_depth_limit + 1):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5) 
        average_score = scores.mean()
        if average_score > best_score:
            best_score = average_score
            best_depth = depth
    

    clf_best = DecisionTreeClassifier(criterion=criterion, max_depth=best_depth, random_state=42)
    clf_best.fit(X, y)
    
    return best_depth, best_score, clf_best


optimal_depth_gini, best_score_gini, clf_best_gini = find_optimal_tree_depth(X, y, max_depth_limit, criterion='gini')
optimal_depth_entropy, best_score_entropy, clf_best_entropy = find_optimal_tree_depth(X, y, max_depth_limit, criterion='entropy')

# Visualize the trees side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

s=['x1','x2','x3','x4','y']

# Plot the first tree
tree.plot_tree(clf_best_gini, filled=True, feature_names=s, class_names=['-1', '1'], ax=axes[0])
axes[0].set_title(f'Best Decision Tree (Depth: {optimal_depth_gini}, Criterion: Gini, Accuracy: {best_score_gini:.2f})')

# Plot the second tree
tree.plot_tree(clf_best_entropy, filled=True, feature_names=s, class_names=['-1', '1'], ax=axes[1])
axes[1].set_title(f'Best Decision Tree (Depth: {optimal_depth_entropy}, Criterion: Entropy, Accuracy: {best_score_entropy:.2f})')

plt.show()