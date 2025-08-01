import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import PolynomialFeatures


train_feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_feature.csv')
train_label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_label.csv')
X_train = train_feature.values
y_train = train_label.values


test_feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_feature.csv')
test_label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_label.csv')
X_test = test_feature.values
y_test = test_label.values


i = 3
poly = PolynomialFeatures(degree=i, include_bias=False)
design_matrix_train = poly.fit_transform(X_train)
design_matrix_test = poly.transform(X_test)
'''
model = Ridge(fit_intercept=True, alpha=0.1)
model.fit(design_matrix_train, y_train)

y_hat_train = model.predict(design_matrix_train)
y_hat_test = model.predict(design_matrix_test)
'''

W = np.linalg.inv(design_matrix_train.T @ design_matrix_train) @ design_matrix_train.T @ y_train

y_hat_train = design_matrix_train @ W
y_hat_test = design_matrix_test @ W


print(np.mean((y_train - y_hat_train) ** 2))
print(np.mean((y_test - y_hat_test) ** 2))