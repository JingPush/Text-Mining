#!/usr/bin/env python2.7
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score


Class LogisticRegression():

	
	def __int__(self):
		self.weight = None


	def learn_LR(self, X, y):
    	w = np.zeros((101, len(X[1, :])))
    	pred_prob = np.zeros((len(X), 1))
    	count = 0
    	tol = 1
    	gradient = np.zeros((len(X[1, :]), 1))
    	while ((count < 100) & (tol > 1e-6)):
        	for i in range(len(X)):
            	pred_prob[i] = 1 / (1 + np.exp(-np.dot(w[count, :], X[i, :])))
        	for j in range(len(gradient)):
            	gradient[j] = 0.01 * w[count, j] - sum(np.multiply((y - pred_prob.T)[0], X[:, j]))
            	w[count + 1, j] = w[count, j] - 0.01 * gradient[j]
        	tol = LA.norm(w[count + 1, :] - w[count, :])
        	count += 1
        self.weight = w[count-1, :]
    	return self.weight



	def pred_LR(self, X):
    	pred_label = np.zeros((X.shape[0], 1))
    	loss = 0
    	for i in range(X.shape[0]):
        	p1 = 1 / (1 + np.exp(-np.dot(self.weight, X[i, :])))
        	if(p1 >= 0.5):
            	pred_label[i] = 1  
 	  	return pred_label



if __name__ == '__main__':
	lr_model = LogisticRegression()
	lr_model.learn_LR(X, y)
	y_hat = lr_model.pred_LR(X)
	train_acc = accuracy_score(y, y_hat)
	print 'ZERO-ONE-LOSS(train)', train_acc





