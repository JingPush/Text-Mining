#!/usr/bin/env python2.7
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score


Class SupportVectorMachine():


	def __init__(self, X):
		self.weight = None
		self.nrow, self.ncol = X.shape


	def svms_vector(self, X):
		vectors = np.zeros((self.nrow, self.ncol+1))
		vectors[:, 0] = 1
		vectors[:, 1:(self.ncol+1)] = X
	return vectors


	def learn_SVMS(self, X, y):
		x = svms_vector(self, X)
		y[y==0] = -1
		w = np.zeros((101, self.ncol+1))
		w[0, 0] = 1
		count = 0
		tol = 1
		gradient = np.zeros((self.ncol+1, 1))
		while ((count < 100) & (tol > 1e-6)):
			for j in range(self.ncol):
				subgraduent = 0
				for i in range(self.nrow):
					y_hat = np.dot(w[count], x[i])
					if (y_hat * y[i] < 1):
						subgraduent += y[i] * x[i, j]
				gradient[j] = 0.01 * w[count, j] - subgradient/float(self.nrow)
				w[count+1, j] = w[count, j] - 0.5 * gradient[j]
			tol = abs(LA.norm(w[count + 1, :]-w[count, :]))
			count += 1
		self.weight = w[count-1, :]



	def pred_SVMS(self, X):
		pred_label = np.zeros((X.shape[0], 1))
		for i in range(X.shape[0]):
			p1 = np.dot(self.weight, X[i, :])
			if(p1 >= 0):
				pred_label[i] = 1
		return pred_label



if __name__ == '__main__':
	svm_model = SupportVectorMachine()
	svm_model.learn_SVMS(X, y)
	y_hat = svm_model.pred_SVMS(X)
	train_acc = accuracy_score(y, y_hat)
	print 'ZERO-ONE-LOSS(train)', train_acc


