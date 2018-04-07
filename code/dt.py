#!/usr/bin/env python2.7
import numpy as np
from numpy import linalg as LA
from collections import Counter
import random
import math
import matplotlib.pyplot as plt
from scipy import stats



Class DecisionTree():


	def __init__(self, X):
		self.nrow, self.ncol = X.shape
		self.tree = None


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



	def split_best_tree(self, dataset):
    	max_gain = 0
    	x = dataset['x']
    	y = dataset['y']
    	best_left, best_right, index = None, None, None
    	for i in range(self.ncol):
        	left_x, left_y, right_x, right_y = split_data(dataset, i)
        	gain = get_gain(y, left_y, right_y)
        	if gain > max_gain:
            	max_gain = gain
            	index = i
            	best_left = {'x': left_x, 'y': left_y}
            	best_right = {'x': right_x, 'y': right_y}
    	return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}



	def to_terminal(self, node):
    	y = node['y']
    	pred_y = 0
    	if np.count_nonzero(y == 1) >= np.count_nonzero(y == 0):
        	pred_y = 1
    	return pred_y



	def split_tree(self, node, depth, max_depth, min_size):

    	if node['pivot'] == None :
        	node['left'] = to_terminal(node)
        	node['right'] = to_terminal(node)
        	return

    	left = node['left']
    	right = node['right']
    	#print len(left['y']), len(right['y']), node['pivot']

    	if depth >= max_depth:
        	node['left'], node['right'] = to_terminal(left), to_terminal(right)
        	return

    	if len(left['y']) <= min_size:
        	node['left'] = to_terminal(left)
        	if len(right['y']) <= min_size:
            	node['right'] = to_terminal(right)
            	return
        	else:
            	node['right'] = split_best_tree(right)
            	split_tree(node['right'], depth+1, max_depth, min_size)
        	return

    	else:
        	node['left'] = split_best_tree(left)
        	split_tree(node['left'], depth+1, max_depth, min_size)

        	if len(right['y']) <= min_size:
            	node['right'] = to_terminal(right)
            	return
        	else:
            	node['right'] = split_best_tree(right)
            	split_tree(node['right'], depth+1, max_depth, min_size)


	def build_tree(self, X, y, max_depth, min_size):
    	trainset = {'x': X, 'y': y}
    	root = split_best_tree(self, trainset)
    	split_tree(self, root, 1, max_depth, min_size)
    	self.tree = root


	def pred(self, x):
		node = self.tree
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


if __name__ == '__main__':
	Tree_model = DecisionTree()
	Tree_model.build_tree(self, X, y, max_depth, min_size)
	y_hat = Tree_model.pred_tree(X)
	train_acc = accuracy_score(y, y_hat)
	print 'ZERO-ONE-LOSS(train)', train_acc








