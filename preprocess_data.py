import numpy as np
from collections import Counter
import csv

class Preprocess_data():


	def __init__(self, data, k_number_of_features=500):
		self.k = k_number_of_features
		self.words = zip(*data)[2]


	def get_word(self, data):
		punc1 = ("~`!@#$%^&*()_-+=[]{}\|;:',<.>/?")
		punc2 = ('"')
		wordsbag = []
		words = zip(*data)[2]
		words = [item.lower().translate(None, punc1).translate(None, punc2) for item in words]
		self.words = [item.split() for item in words]
		for line in self.words:
			wordsbag.extend(set(line))
		return wordsbag


	def count_attr(self,data):
		c = Counter(self.get_word(data))
		feature = c.most_common(100+self.k)[100:100+self.k]
		return feature


	def summarize_feature(self, data):
		words = self.words
		feature = self.count_attr(data)
		feature_value = np.zeros((len(data), len(feature)))
		for i in range(len(words)):
			for j in range(len(feature)):
				if (feature[j][0] in words[i]):
					feature_value[i][j] = 1
				else:
					feature_value[i][j] = 0
		return feature_value



if __name__=='__main__':
	file = open('csv_file', 'rU')
	data = list(csv.reader(file, delimiter='\t'))
	preprocessed = Preprocess_data(data, k_number_of_features=500)
	wordsbag = preprocessed.get_word(data)
	feature = preprocessed.count_attr(data)
	feature_value = preprocessed.summarize_feature(data)
	#-------print the most common ten words---------#
	for i in range(10):
		print 'WORD' + str(i+1), feature[i][0]


