#!/usr/bin/env python2.7
import numpy as np
from numpy import linalg as LA
from collections import Counter
import random
import math
import matplotlib.pyplot as plt
from scipy import stats



Class AdaboostTree():



	def __init__(self, X, num, max_depth, min_size):
		self.nrow, self.ncol = X.shape
		self.alpha = None
		self.tree = None
		self.num = num
		self.max_depth = max_depth
		self.min_size = min_size



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


	def boost_gini(y, w):
	    gini = 0
	    if len(y) > 0:
	        sum_total = sum(w)
	        sum_w_1 = sum(w[y == 1])
	        sum_w_0 = sum(w[y == 0])
	        gini = 1 - (sum_w_1 / sum_total)**2 - (sum_w_0 / sum_total)**2
	    return gini


	def boost_gain(upper_y, upper_w, left_y, left_w, right_y, right_w):
	    gain_left = boost_gini(left_y, left_w) * sum(left_w) / sum(upper_w)
	    gain_right = boost_gini(right_y, right_w) * sum(right_w) / sum(upper_w)
	    gain = boost_gini(upper_y, upper_w) - gain_left - gain_right
	    return gain



	def split_boost_data(dataset, index):
	    x = dataset['x']
	    y = dataset['y']
	    w = dataset['w']
	    left_x = x[x[:, index] == 1]
	    left_y = y[x[:, index] == 1]
	    left_w = w[x[:, index] == 1]
	    right_x = x[x[:, index] == 0]
	    right_y = y[x[:, index] == 0]
	    right_w = w[x[:, index] == 0]
	    return left_x, left_y, left_w, right_x, right_y, right_w



	def split_best_boost(self, dataset):
	    max_gain = 0
	    x = dataset['x']
	    y = dataset['y']
	    w = dataset['w']
	    best_left, best_right, index = None, None, None
	    for i in range(self.ncol):
	        left_x, left_y, left_w, right_x, right_y, right_w = split_boost_data(dataset, i)
	        gain = boost_gain(y, w, left_y, left_w, right_y, right_w)
	        if gain > max_gain:
	            max_gain = gain
	            index = i
	            best_left = {'x': left_x, 'y': left_y, 'w': left_w}
	            best_right = {'x': right_x, 'y': right_y, 'w': right_w}
	    return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}



	def split_tree(self, node, depth):

	    if node['pivot'] == None :
	        node['left'] = to_terminal(node)
	        node['right'] = to_terminal(node)
	        return

	    left = node['left']
	    right = node['right']
	    #print len(left['y']), len(right['y']), node['pivot']

	    if depth >= self.max_depth:
	        node['left'], node['right'] = to_terminal(left), to_terminal(right)
	        return

	    if len(left['y']) <= self.min_size:
	        node['left'] = to_terminal(left)
	        if len(right['y']) <= self.min_size:
	            node['right'] = to_terminal(right)
	            return
	        else:
	            node['right'] = split_best_boost(right)
	            split_boost_tree(self, node['right'], depth+1)
	        return

	    else:
	        node['left'] = split_best_boost(left)
	        split_boost_tree(self, node['left'], depth+1)

	        if len(right['y']) <= self.min_size:
	            node['right'] = to_terminal(right)
	            return
	        else:
	            node['right'] = split_best_boost(right)
	            split_boost_tree(self, node['right'], depth+1)



	def build_boost_tree(self, X, y, w):
	    train = {'x': X, 'y': class_label, 'w': w}
	    root = split_best_boost(self, train)
	    split_boost_tree(self, root, 1)
	    return root



	def train_boost(self, X, y):
	    w = np.full((self.nrow, self.num+1), 1/float(self.nrow))
	    alpha = np.zeros((self.num, 1))
	    total_tree = []
	    for t in range(self.num):
	        w_temp = w[:, t]
	        tree = build_boost_tree(self, X, y, w_temp)
	        total_tree.append(tree)
	        pred_y = pred_tree(tree, X)
	        pred_y[pred_y == 0] = -1
	        sum_error = sum(w_temp[(true_y != pred_y.T)[0]])
	        if sum_error == 0:
	            alpha[t] = 100
	            break
	        alpha[t] = 0.5 * math.log((1 - sum_error) / sum_error)
	        if alpha[t] == 0:
	            break
	        for i in range(self.nrow):
	            w[i, t+1] = w_temp[i] * math.exp(- alpha[t] * true_y[i] * pred_y[i])
	        w[:, t+1] = w[:, t+1]/sum(w[:, t+1])
	    self.alpha = alpha
	    self.tree = total_tree




	def pred_boost(self, X):
	    pred_total = np.zeros(X.shape[0], self.num)
	        for i in range(len(self.tree)):
        		tree = self.tree[i]
        		pred_total[:, i] = pred_tree(tree, X)[:, 0]
    	pred_final = np.zeros(X.shape[0], 1)
    	for i in range(len(test)):
        	pred_final[i] = np.dot(alpha[:, 0], pred_total[i, :])
    	return pred_final




if __name__ == '__main__':
	Tree_model = AdaboostTree(X, num, max_depth, min_size)
	Tree_model.train_boost(X, y)
	y_hat = Tree_model.pred_boost(X)
	train_acc = accuracy_score(y, y_hat)
	print 'ZERO-ONE-LOSS(train)', train_acc



