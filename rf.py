import numpy as np
from numpy import linalg as LA
from collections import Counter
import random
import math
import matplotlib.pyplot as plt
from scipy import stats



Class Randomforest():



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



	def to_terminal(self, node):
    	y = node['y']
    	pred_y = 0
    	if np.count_nonzero(y == 1) >= np.count_nonzero(y == 0):
        	pred_y = 1
    	return pred_y



	def split_best_forest(self, k):
        y = dataset['y']
        x = dataset['x']
        max_gain = 0
        best_left, best_right, index = None, None, None
        select_index = random.sample(list(range(len(x[0, :]))), k)
        for i in range(k):
            m = select_index[i]
            left_x, left_y, right_x, right_y = split_data(dataset, m)
            gain = get_gain(y, left_y, right_y)
            if gain > max_gain:
                max_gain = gain
                index = m
                best_left = {'x': left_x, 'y': left_y}
                best_right = {'x': right_x, 'y': right_y}
        return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}
    


    def split_RF(self, node, depth, max_depth, min_size, k):

        if node['pivot'] == None :
            node['left'] = to_terminal(node)
            node['right'] = to_terminal(node)
            return

        left = node['left']
        right = node['right']

        if depth >= max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return

        if len(left['y']) <= min_size:
            node['left'] = to_terminal(left)
            if len(right['y']) <= min_size:
                node['right'] = to_terminal(right)
                return
            else:
                node['right'] = split_best_forest(right, k)
                split_RF(node['right'], depth+1, max_depth, min_size, k)
            return

        else:
            node['left'] = split_best_forest(left, k)
            split_RF(node['left'], depth+1, max_depth, min_size, k)

            if len(right['y']) <= min_size:
                node['right'] = to_terminal(right)
                return
            else:
                node['right'] = split_best_forest(right, k)
                split_RF(node['right'], depth+1, max_depth, min_size, k)



    def build_RF(self, train, max_depth, min_size, feature, k):
        x = summarize_feature(train, feature)
        class_label = np.asarray([int(item) for item in zip(*train)[1]])
        train = {'x': x, 'y': class_label}
        root = split_best_forest(train, k)
        split_RF(root, 1, max_depth, min_size, k)
        return root



    def pred_RF(self, train, test, num, max_depth, min_size):
        pred_total = np.zeros((len(test), num))
        pred_final = np.zeros((len(test), 1))
        feature = count_attr(train)
        k = int(math.sqrt(len(feature)))
        for i in range(num):
            new_train = [random.choice(train) for j in range(len(train))]
            tree = build_RF(new_train, max_depth, min_size, feature, k)
            pred_total[:, i] = pred_tree(tree, feature, test)[:, 0]
        for i in range(len(test)):
            if np.count_nonzero(pred_total[i, :]) >= num/2:
                pred_final[i] = 1
        return pred_final



if __name__ == '__main__':
	RF_model = Randomforest()
	y_hat = RF_model.pred_BT(self, X, num, max_depth, min_size)
    train_acc = accuracy_score(y, y_hat)
    print 'ZERO-ONE-LOSS(train)', train_acc


