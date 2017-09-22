# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       tree
   Description: 
   Author:          rhys
   date:            17/9/19
-------------------------------------------------
"""

from collections import Counter
import numpy as np


def calc_shannon_ent(y):
    """
    calculate shannon entropy
    :param y: the labels
    :return: shannon entropy
    """
    m = y.shape[0]
    counter = Counter(y)
    label_cnt = np.array(counter.most_common())[:, 1]
    pros = label_cnt / m
    return -np.sum(pros * np.log2(pros))


def split_data_set(X, axis, value):
    """
    find out indexes of X where X[:, axis] == value
    :param X:
    :param axis:
    :param value:
    :return: the indexes
    """
    ret = []
    for idx, x in enumerate(X):
        if x[axis] == value:
            ret.append(idx)
    return np.array(ret)


def choose_feature(X, y):
    """
    travel around all features and choose the best feature(which get maximum information gain)
    :param X: data set
    :param y: label
    :return: feature index
    """
    base_ent = calc_shannon_ent(y)
    num_features = X.shape[1]

    best_gain = 0.
    best_feature = -1
    for i in range(num_features):
        feature_val = np.unique(X[:, i].copy())
        new_ent = 0.0
        for val in feature_val:
            sub_data = split_data_set(X, i, val)
            prob = sub_data.shape[0] / X.shape[0]
            new_ent += prob * calc_shannon_ent(np.take(y, sub_data))
        gain = base_ent - new_ent

        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    counter = Counter(class_list)
    return counter.most_common(1)[0][0]


def create_tree(X, y, labels):

    if len(np.unique(y)) == 1:
        return y[0]
    if len(X[0]) == 0:
        return majority_cnt(y)
    best_feat = choose_feature(X, y)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]

    for val in np.unique(X[:, best_feat]):
        sub_labels = labels[:]
        sub_dataset_idxs = split_data_set(X, best_feat, val)
        sub_X = X[sub_dataset_idxs, :]
        sub_X = np.delete(sub_X, best_feat, 1)
        sub_y = y[sub_dataset_idxs]
        my_tree[best_feat_label][val] = create_tree(sub_X, sub_y, sub_labels)
    return my_tree


def classify(tree, labels, test_vec):

    if type(tree).__name__ != 'dict':
        return tree
    key = list(tree.keys())[0]

    feature_idx = labels.index(key)

    return classify(tree[key][test_vec[feature_idx]], labels, test_vec)


if __name__ == '__main__':
    labels = ['no surfacing', 'flippers']
    X = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y = np.array([1, 1, 0, 0, 0])
    r = create_tree(X, y, labels.copy())
    print(r)

    print(classify(r, labels, [1, 0]))
