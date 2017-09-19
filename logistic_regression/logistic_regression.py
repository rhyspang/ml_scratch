# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       logistic_regression
   Description: 
   Author:          rhys
   date:            17/9/13
-------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def cost(theta, X, y):
    h = h_func(theta, X)
    return np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))


def h_func(theta, X):
    return sigmoid(X @ theta)


def gradient(theta, X, y):
    return X.T @ (h_func(theta, X) - y) / X.shape[0]


if __name__ == '__main__':
    df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    X = df[['exam1', 'exam2']].as_matrix()
    y = df[['admitted']].as_matrix().flatten()

    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
    print(res)
    final_theta = res.x

    coef = -(res.x / res.x[2])
    x = np.arange(130, step=0.1)
    y = coef[0] + coef[1]*x

    not_admitted = df[df['admitted'] == 1]
    admitted = df[df['admitted'] == 0]
    plt.plot(not_admitted['exam1'], not_admitted['exam2'], 'rx', admitted['exam1'], admitted['exam2'], 'bo')
    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.grid(True)
    plt.plot(x, y, 'grey')
    plt.show()