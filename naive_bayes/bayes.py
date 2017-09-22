# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       bayes
   Description: 
   Author:          rhys
   date:            17/9/21
-------------------------------------------------
"""

import numpy as np


def load_data():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def words2vec(voca_list, input_set):
    ret_vec = [0] * len(voca_list)
    for word in input_set:
        if word in voca_list:
            ret_vec[voca_list.index(word)] = 1
        else:
            print('the word: %s is not in vocabulary list')
    return ret_vec


def trainNB0(X, y):
    m = X.shape[0]
    num_words = X.shape[1]

    p_abusive = np.sum(y) / m
    p0_num, p1_num = np.ones(num_words), np.ones(num_words)
    p0_denom, p1_denom = 2.0, 2.0

    for i in range(m):
        if y[i] == 1:
            p1_num += X[i]
            p1_denom += np.sum(X[i])
        else:
            p0_num += X[i]
            p0_denom += np.sum(X[i])

    print('p1_denom: %s, p0_denom: %s' % (p1_denom, p0_denom))
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive


def classifyNB(vec, p0v, p1v, p_class1):
    p1 = np.sum(vec*p1v) + np.log(p_class1)
    p0 = np.sum(vec*p0v) + np.log(1-p_class1)
    return p1 if p1 > p0 else p0


def 


if __name__ == '__main__':
    posts, classes = load_data()
    vocab_list = create_vocab_list(posts)
    X = []
    for item in posts:
        X.append(words2vec(vocab_list, item))
    X = np.array(X)
    classes = np.array(classes)
    print(X.shape)
    print(X)
    print(classes)

    p0_v, p1_v, p_ab = trainNB0(X, classes)
    print(p0_v, p1_v, p_ab, sep='\n\n')

