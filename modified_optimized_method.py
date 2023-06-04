import datetime

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error

from sklearn.preprocessing import MaxAbsScaler
import math
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, ClassifierMixin


class GRNN(BaseEstimator, ClassifierMixin):
    def __init__(self, name="GRNN", sigma=0.1):
        self.name = name
        self.sigma = 2 * np.power(sigma, 2)

    def predict(self, instance_X, train_X, train_y):
        gausian_distances = np.exp(-np.power(np.sqrt((np.square(train_X - instance_X).sum(axis=1))), 2) \
                                   / self.sigma)
        gausian_distances_sum = gausian_distances.sum()
        if gausian_distances_sum < math.pow(10, -7): gausian_distances_sum = math.pow(10, -7)
        result = np.multiply(gausian_distances, train_y).sum() / gausian_distances_sum
        return result


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


df_train = pd.read_csv("body_fat_train.txt", header=None)
df_test = pd.read_csv("body_fat_test.txt", header=None)

train_X = df_train.iloc[:, :-1]
train_y = df_train.iloc[:, -1]
test_X = df_test.iloc[:, :-1]
test_y = df_test.iloc[:, -1]

scaler = MaxAbsScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)


def f(s):
    train_y_predictions = []
    for i in range(0, len(train_X)):
        grnn = GRNN(sigma=s[0])
        train_X_1 = train_X.copy()
        train_y_1 = train_y.copy()
        train_X_1 = np.delete(train_X_1, obj=i, axis=0)
        train_y_1 = train_y_1.drop([i])
        train_y_predictions.append(grnn.predict(train_X[i], train_X_1, train_y_1))

    test_y_predictions = []
    for i in range(0, len(test_X)):
        grnn = GRNN(sigma=s[0])
        test_y_predictions.append(grnn.predict(test_X[i], train_X, train_y))

    train_X_augmented = np.empty([len(train_X) ** 2, len(train_X[0]) * 2 + 2])
    train_Z_augmented = []
    counter = 0
    for i in range(0, len(train_X)):
        for j in range(0, len(train_X)):
            train_X_augmented[counter] = np.append(np.append(np.concatenate([train_X[i], train_X[j]]),
                                                             train_y[i]),
                                                   train_y[j])
            train_Z_augmented.append(train_y[i] - train_y[j])
            counter += 1
    train_Z_augmented = pd.Series(train_Z_augmented)
    test_Z_augmented = [[0 for _ in range(0, len(train_X))] for _ in range(0, len(test_X))]
    for i in range(0, len(test_X)):
        for j in range(0, len(train_X)):
            grnn = GRNN(sigma=s[1])
            test_Z_augmented[i][j] = grnn.predict(np.append(np.append(np.concatenate([test_X[i], train_X[j]]),
                                                                      test_y_predictions[i]),
                                                            train_y[j]), train_X_augmented,
                                                  train_Z_augmented)

    test_y_final_predictions = []
    for i in range(0, len(test_Z_augmented)):
        test_y_final_predictions.append((sum(test_Z_augmented[i]) + sum(train_y)) / len(train_y))
    test_y_final_predictions = pd.Series(test_y_final_predictions)
    print("Sigma1 = " + str(s[0]))
    print("Sigma2 = " + str(s[1]))
    print('Testing errors:')
    print("MAPE: " + str(mean_absolute_percentage_error(test_y, test_y_final_predictions)))
    print("RMSE: " + str(root_mean_squared_error(test_y, test_y_final_predictions)))
    print("MAE: " + str(mean_absolute_error(test_y, test_y_final_predictions)))
    print("R2: " + str(r2_score(test_y, test_y_final_predictions)))
    print("MSE: " + str(mean_squared_error(test_y, test_y_final_predictions)))
    print("ME: " + str(max_error(test_y, test_y_final_predictions)))
    print("MedAE: " + str(median_absolute_error(test_y, test_y_final_predictions)))
    return -r2_score(test_y, test_y_final_predictions)


start = datetime.datetime.utcnow()
res = differential_evolution(f, bounds=[(0.01, 10), (0.01, 10)], strategy='randtobest1bin')
end = datetime.datetime.utcnow()
print("Time: " + str((end - start).total_seconds()))
