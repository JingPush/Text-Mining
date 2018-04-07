#!/usr/bin/env python2.7
import numpy as np
from numpy import linalg as LA
from collections import Counter
import random
import math
import matplotlib.pyplot as plt
from scipy import stats



Class Randomforest():



	def __init__(self, X, num, max_depth, min_size):
		self.nrow, self.ncol = X.shape
		self.tree = None
        self.max_depth = max_depth
        self.min_size = min_size
        self.num = num
        self.k = int(math.sqrt(self.ncol))



    def pred(node, x):
        if node['pivot'] == None:
            return node['left']
        if x[node['pivot']] == 1:
            if isinstance(node['left'], dict):
                return pred(node['left'], x)
            else:
                return node['left']
        if x[node['pivot']] == 0:
            if isinstance(node['right'], dict):
                return pred(node['right'], x)
            else:
                return node['right']


    def pred_tree(self, X):
        pred_y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            pred_y[i] = pred(X[i, :])
        return pred_y




	def get_gini(self, y):
    	gini = 0
    	length = float(self.nrow)
    	if len(y) > 0:
        	gini = 1 - (np.count_nonzero(y == 0)/length)**2 - (np.count_nonzero(y == 1)/length)**2
    	return gini



	def get_gain(self, upper, left, right):
    	gain = get_gini(upper) - get_gini(left) * len(left)/len(upper) - get_gini(right) * len(right)/len(upper)
    	return gain



	def split_data(self, dataset, index):
    	x = dataset['x']
    	y = dataset['y']
    	left_x = x[x[:, index] == 1]
    	left_y = y[x[:, index] == 1]
    	right_x = x[x[:, index] == 0]
    	right_y = y[x[:, index] == 0]
    	return left_x, left_y, right_x, right_y



	def to_terminal(self, node):
    	y = node['y']
    	pred_y = 0
    	if np.count_nonzero(y == 1) >= np.count_nonzero(y == 0):
        	pred_y = 1
    	return pred_y



	def split_best_forest(self, dataset):
        y = dataset['y']
        x = dataset['x']
        max_gain = 0
        best_left, best_right, index = None, None, None
        select_index = random.sample(list(range(self.nrow), self.k)
        for i in range(self.k):
            m = select_index[i]
            left_x, left_y, right_x, right_y = split_data(self, dataset, m)
            gain = get_gain(y, left_y, right_y)
            if gain > max_gain:
                max_gain = gain
                index = m
                best_left = {'x': left_x, 'y': left_y}
                best_right = {'x': right_x, 'y': right_y}
        return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}
    


    def split_RF(self, node, depth):

        if node['pivot'] == None :
            node['left'] = to_terminal(node)
            node['right'] = to_terminal(node)
            return

        left = node['left']
        right = node['right']

        if depth >= self.max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return

        if len(left['y']) <= self.min_size:
            node['left'] = to_terminal(left)
            if len(right['y']) <= self.min_size:
                node['right'] = to_terminal(right)
                return
            else:
                node['right'] = split_best_forest(self, right)
                split_RF(self, node['right'], depth+1)
            return

        else:
            node['left'] = split_best_forest(self, left)
            split_RF(self, node['left'], depth+1)

            if len(right['y']) <= self.min_size:
                node['right'] = to_terminal(right)
                return
            else:
                node['right'] = split_best_forest(self, right)
                split_RF(self, node['right'], depth+1)



    def build_RF(self, X, y):
        train = {'x': X, 'y': y}
        root = split_best_forest(self, train)
        split_RF(self, root, 1)
        return root



    def train_RF(self, X, y):
        pred_total = np.zeros((self.nrow, self.num))
        pred_final = np.zeros((self.nrow, 1))
        train_index = range(self.nrow)
        self.tree = None
        for i in range(self.num):
            index = [random.choice(train_index) for j in range(self.nrow)]
            X_new = X[index, :]
            y_new = y[index]
            tree = build_RF(self, X_new, y_new)
            self.tree.append(tree)



    def pred_RF(self, X):
        for i in range(self.num):
            pred_total[:, i] = pred_tree(self.tree[i], X)[:, 0]
        for i in range(X.shape[0]):
            if np.count_nonzero(pred_total[i, :]) >= self.num/2:
                pred_final[i] = 1
        return pred_final



if __name__ == '__main__':
	RF_model = Randomforest(X, num, max_depth, min_size)
    RF_model.train_RF(X, y)
    y_hat = RF_model.pred_RF(X)
    train_acc = accuracy_score(y, y_hat)
    print 'ZERO-ONE-LOSS(train)', train_acc


