# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       show
   Description: 
   Author:          rhys
   date:            17/9/13
-------------------------------------------------
"""

import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    print(df.shape)
    print(df.head())
    not_admitted = df[df['admitted'] == 1]
    admitted = df[df['admitted'] == 0]
    plt.plot(not_admitted['exam1'], not_admitted['exam2'], 'rx', admitted['exam1'], admitted['exam2'], 'bo')
    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
