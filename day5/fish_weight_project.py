import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error

train_feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_feature.csv')
train_label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_label.csv')
X_train = train_feature.values
y_train = train_label.values


test_feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_feature.csv')
test_label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_label.csv')
X_test = test_feature.values
y_test = test_label.values

i = 2

poly = PolynomialFeatures(degree=i, include_bias=False)
design_matrix_train = poly.fit_transform(X_train)
design_matrix_test = poly.transform(X_test)

model = Ridge(fit_intercept=True, alpha=0.1)
model.fit(design_matrix_train, y_train)

y_hat_train = model.predict(design_matrix_train)
y_hat_test = model.predict(design_matrix_test)

print(mean_squared_error(y_train, y_hat_train))
print(mean_squared_error(y_test, y_hat_test))

labels = ["Length1", "Length2", "Length3", "Height", "Width"]

for i in range(5):
    X_single = X_train[:, i].reshape(-1, 1)

    # Train model on just one feature (if you haven't already)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    design_matrix_train = poly.fit_transform(X_single)   

    model = Ridge(fit_intercept=True, alpha=0.1)
    model.fit(design_matrix_train, y_train)

    # Create line
    X_line1 = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
    design_line = poly.fit_transform(X_line1)
    y_line1 = model.predict(design_line)

    plt.figure()
    plt.plot(X_line1, y_line1)
    plt.scatter(X_test[:, i], y_test)
    plt.xlabel(f'{labels[i]}')
    plt.ylabel('Weight')
    plt.title(f'{labels[i]} vs Weight')
    plt.ylim(-200, 1400)
    plt.show()



