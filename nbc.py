#!/usr/bin/env python2.7
import numpy as np
from scipy.stats import itemfreq
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification



Class NBClassifier():


	def __init__(self, X):
		self.nrow, self.ncol = X.shape
		self.cpds = None
		self.prior = None


	def get_frequency(self, X, y):
		count_feature_1 = np.zeros((self.col,2))
		count_feature_0 = np.zeros((self.col,2))
		for i in range(self.nrow):
			for j in range(self.ncol):
				if (X[y[i]==1, j] == 1):
					count_feature_1[j,0] += 1
				if (X[y[i]==1, j] == 0):
					count_feature_1[j,1] += 1
				if (X[y[i]==0, j] == 1):
					count_feature_1[j,0] += 1
				if (X[y[i]==0, j] == 0):
					count_feature_1[j,1] += 1
		count_feature = np.concatenate((count_feature_1 , count_feature_0),axis = 1)
		return count_feature



	def get_cpds(self, X, y):
		count_feature = get_frequency(X, y)
		count = [np.count_nonzero(y == 1),np.count_nonzero(y == 0)]
		prior_yes = float(count[0])/float(self.nrow)
		prior_no = float(count[1])/float(self.nrow)
		prior = [prior_yes,prior_no]
		cpds_smoothed = np.zeros((len(count_feature), 4))
		for i in range(len(count_feature)):
			cpds_smoothed[i, 0] = float(count_feature[i, 0] + 1) / float(count_feature[i, 0] + count_feature[i, 1] + 2)
			cpds_smoothed[i, 1] = float(count_feature[i, 1] + 1) / float(count_feature[i, 0] + count_feature[i, 1] + 2)
			cpds_smoothed[i, 2] = float(count_feature[i, 2] + 1) / float(count_feature[i, 2] + count_feature[i, 3] + 2)
			cpds_smoothed[i, 3] = float(count_feature[i, 3] + 1) / float(count_feature[i, 2] + count_feature[i, 3] + 2)
		self.cpds = cpds_smoothed
		self.prior = prior
		return [cpds_smoothed , prior]


	def predict(self, X):
		prob_yes = np.full((len(X), 1), self.prior[0])
		prob_no = np.full((len(X), 1), self.prior[1])
		prediction = np.zeros((len(X), 1))
		for i in range(len(X)):
			for j in range(len(X[i,:])):
				if(X[i, j] == 1):
					prob_yes[i] = prob_yes[i] * self.cpds[j, 0]
					prob_no[i] = prob_no[i] * self.cpds[j, 2]
				else:
					prob_yes[i] = prob_yes[i] * self.cpds[j, 1]
					prob_no[i] = prob_no[i] * self.cpds[j, 3]
			if (prob_yes[i] > prob_no[i]):
				prediction[i] = 1
			else:
				prediction[i] = 0
		return prediction



if __name__=='__main__':
	nbc = NBClassifier()
	y_hat = nbc.predict(X)
	train_acc = accuracy_score(y, y_hat)
	print 'ZERO-ONE-LOSS(train)', train_acc








