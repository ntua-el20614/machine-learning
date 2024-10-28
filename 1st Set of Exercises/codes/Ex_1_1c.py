import numpy as np
import pandas as pd


data = pd.read_csv('..\\ML2023-24-hwk1.csv', delimiter=';')
training_set = data.iloc[:100, :]
verification_set = data.iloc[100:150, :]


X_train = training_set.iloc[:, :-1].values
y_train = training_set.iloc[:, -1].values


means = np.mean(X_train, axis=0)
stds = np.std(X_train, axis=0)

stds[stds == 0] = 1

X_train_normalized = (X_train - means) / stds


X_train_normalized = np.concatenate([np.ones((X_train_normalized.shape[0], 1)), X_train_normalized], axis=1)


def calculate_ridge_weights(X, y, lambd):
    identity_matrix = np.eye(X.shape[1])
    identity_matrix[0, 0] = 0
    ridge_weights = np.linalg.inv(X.T.dot(X) + lambd * identity_matrix).dot(X.T).dot(y)
    return ridge_weights


lambdas = [10, 100, 200]


ridge_weights_dict = {lambd: calculate_ridge_weights(X_train_normalized, y_train, lambd) for lambd in lambdas}


for lambd, weights in ridge_weights_dict.items():
    print(f"For λ={lambd}:\n")
    print("    Bias (Intercept): {:.4f}".format(weights[0]))
    for i, weight in enumerate(weights[1:], start=1):
        print(f"    {training_set.columns[i-1]}: {weight:.4f}")
    print("\n")

