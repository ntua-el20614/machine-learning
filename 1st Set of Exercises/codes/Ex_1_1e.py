import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def standardize_features(X, mean, std):
    return (X - mean) / std


def ridge_regression_weights(X, y, lambd):
    identity_matrix = np.eye(X.shape[1])
    identity_matrix[0, 0] = 0  
    return np.linalg.inv(X.T.dot(X) + lambd * identity_matrix).dot(X.T).dot(y)


def predict(X, weights):
    return X.dot(weights)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


data = pd.read_csv('../ML2023-24-hwk1.csv', delimiter=';')
training_set = data.iloc[:100, :]
verification_set = data.iloc[100:150, :]


X_train = training_set.iloc[:, :-1].values
y_train = training_set.iloc[:, -1].values
X_test = verification_set.iloc[:, :-1].values
y_test = verification_set.iloc[:, -1].values


train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)


X_train_std = standardize_features(X_train, train_mean, train_std)
X_test_std = standardize_features(X_test, train_mean, train_std)


X_train_std_with_bias = np.hstack([np.ones((X_train_std.shape[0], 1)), X_train_std])
X_test_std_with_bias = np.hstack([np.ones((X_test_std.shape[0], 1)), X_test_std])


linear_weights = ridge_regression_weights(X_train_std_with_bias, y_train, 0)


lambdas = [10, 100, 200]
ridge_weights = {lambd: ridge_regression_weights(X_train_std_with_bias, y_train, lambd) for lambd in lambdas}


rmse_train_linear = rmse(y_train, predict(X_train_std_with_bias, linear_weights))
rmse_test_linear = rmse(y_test, predict(X_test_std_with_bias, linear_weights))


rmse_train_ridge = {}
rmse_test_ridge = {}
for lambd, weights in ridge_weights.items():
    rmse_train_ridge[lambd] = rmse(y_train, predict(X_train_std_with_bias, weights))
    rmse_test_ridge[lambd] = rmse(y_test, predict(X_test_std_with_bias, weights))

print("RMSE for Linear Regression on training set:", rmse_train_linear)
print("RMSE for Linear Regression on test set:", rmse_test_linear)

print("\nRMSE for Ridge Regression on training and test sets for each lambda:")
for lambd, rmse_train in rmse_train_ridge.items():
    rmse_test = rmse_test_ridge[lambd]
    print(f"For lambda={lambd}:\n  - Training set: {rmse_train}\n  - Test set: {rmse_test}\n")

