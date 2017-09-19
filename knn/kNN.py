# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       kNN
   Description: 
   Author:          rhys
   date:            17/9/18
-------------------------------------------------
"""

import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def create_dataset():
    group = np.array([[1., 1.1], [1., 1.], [0, 0], [0, .1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def classify0(in_x, dataset, labels, k):
    print(dataset.shape, labels.shape)
    n = dataset.shape[0]
    diss = (((np.tile(in_x, (n, 1)) - dataset) ** 2).sum(axis=1)) ** 0.5
    sorted_idxs = diss.argsort()
    counter = Counter()
    for i in range(k):
        counter.update(str(labels[sorted_idxs[i]]))
    return int(counter.most_common(1)[0][0])


def load_datafile(filename):
    df = pd.read_csv(filename, sep='\t', names=['fly_miles', 'game_time', 'ice_cream', 'attitude'])
    X = df[['fly_miles', 'game_time', 'ice_cream']].as_matrix()
    y = df['attitude'].map({'didntLike': 0, 'smallDoses': 1, 'largeDoses': 2}).as_matrix()
    return X, y


def show_data():
    X, y = load_datafile('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], 15 * y + 15, 15 * y + 15)
    plt.xlabel('fly miles')
    plt.ylabel('game time')
    plt.show()


def norm(X):
    min_vals = X.min(0)
    max_vals = X.max(0)
    ranges = max_vals - min_vals
    norm_X = np.zeros_like(X)
    m = X.shape[0]
    norm_X = X - np.tile(min_vals, (m, 1))
    norm_X = norm_X / np.tile(ranges, (m, 1))
    return norm_X, ranges, min_vals


def dating_class_test(ho_ratio=.1):
    X, y = load_datafile('datingTestSet.txt')
    norm_X, _, _ = norm(X)
    m = X.shape[0]

    text_size = int(ho_ratio * m)
    err_cnt = 0
    for i in range(text_size):
        result = classify0(norm_X[i, :], norm_X[text_size:, :], y[text_size:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (result, y[i]))
        err_cnt += int(result != y[i])
    print(err_cnt / float(text_size))


if __name__ == '__main__':
    dating_class_test()
