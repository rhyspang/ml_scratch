# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       linear_regression
   Description: 
   Author:          rhys
   date:            17/9/11
-------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):

    def __init__(self, n_iters, eta):
        self.n_iters = n_iters
        self.eta = eta

    def gradient(self, X, y):
        return X.T @ (X @ self.theta - y) / X.shape[0]

    def cost(self, X, y):
        m = X.shape[0]
        t = X @ self.theta - y
        return t.T @ t / (m << 1)

    def fit(self, X, y):
        """
        fit linear model
        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
                    Training data
        :param y: numpy array of shape [n_samples, n_targets]
                    Target values
        :return:
        """
        X = np.c_[np.ones(X.shape[0]), X]   # add a column 1 in the front of X
        self.theta = np.zeros(X.shape[1])
        cost_list = [self.cost(X, y)]
        for _ in range(self.n_iters):
            self.theta -= self.eta * self.gradient(X, y)
            cost_list.append(self.cost(X, y))
        return self.theta, cost_list


if __name__ == '__main__':
    model = LinearRegression(500, 1e-2)

    a = np.arange(4).reshape(4, 1)
    b = (a*2 + 3).reshape(4,)

    theta, cost = model.fit(a, b)
    print('theta:', theta)
    print('cost: ', cost)
    plt.plot(cost)
    plt.show()
