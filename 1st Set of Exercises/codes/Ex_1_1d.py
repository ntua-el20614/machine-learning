import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear_regression_weights(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def ridge_regression_weights(X, y, lambd):
    identity_matrix = np.eye(X.shape[1])
    identity_matrix[0, 0] = 0
    return np.linalg.inv(X.T.dot(X) + lambd * identity_matrix).dot(X.T).dot(y)


data = pd.read_csv('..\\ML2023-24-hwk1.csv', delimiter=';')
training_set = data.iloc[:100, :]
verification_set = data.iloc[100:150, :]


X_train = training_set.iloc[:, :-1].values  
y_train = training_set.iloc[:, -1].values 


X_train_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)


X_train_normalized = np.hstack([np.ones((X_train_normalized.shape[0], 1)), X_train_normalized])


linear_weights = linear_regression_weights(X_train_normalized, y_train)


lambdas = [10, 100, 200]


ridge_weights = {lambd: ridge_regression_weights(X_train_normalized, y_train, lambd) for lambd in lambdas}


all_weights = {'Linear': linear_weights}
all_weights.update(ridge_weights)


weights_df = pd.DataFrame(all_weights).T
feature_names = ['Bias'] + list(data.columns[:-1])  
weights_df.columns = feature_names


plt.figure(figsize=(15, 10))


for method, weight_vector in weights_df.iterrows():
    plt.plot(weight_vector, label=method)

plt.title('Comparison of Weights: Linear vs Ridge Regression')
plt.xlabel('Features')
plt.ylabel('Weight Value')
plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=45)
plt.legend()
plt.grid(True)


weights_comparison_file_path = '../Plots/weights_comparison_plot.png'
plt.savefig(weights_comparison_file_path)
plt.close()  
weights_comparison_file_path
