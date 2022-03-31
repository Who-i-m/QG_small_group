#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg   #导入numpy中计算线性代数的模块linalg


class Liner_Regression(object):
    def __init__(self):
        self.coef = None  # 代表的是权重
        self.interception = None  # 代表的是截距
        self._theta = None  # 代表的是权重加截距形成的矩阵

    def fit(self, x_train, y_train):
        ones = np.ones((x_train.shape[0], 1))
        x_b = np.hstack((ones, x_train))  
        # 将训练矩阵转换为X_b矩阵，其中第一列为1，其余不变
        # 根据正规方程公式求得权重矩阵，公式：_theta = (X^T * X)^{-1} * X^T * y
        self._theta = linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.interception = self._theta[0]
        self.coef = self._theta[1:]

        return self

    def predict(self, x_test):
        ones = np.ones((x_test.shape[0], 1))
        x_b = np.hstack((ones, x_test))
        return x_b.dot(self._theta)

    def mean_squard_error(self, y_true, y_test):
        return np.sum((y_true - y_test) ** 2) / len(y_true)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return 1 - (self.mean_squard_error(y_test, y_predict) / np.var(y_test))

