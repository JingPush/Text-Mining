# Text-Mining
Boosted Yelp Review Classification     (dataset is from Kaggle: https://www.kaggle.com/yelp-dataset/yelp-dataset)


This project challenges the Yelp Reviews data, and completes the classification task to 5,200,000 user reviews. What's in a review? Is it positive or negative? Yelp's reviews contain a lot of metadata that can be mined and used to infer meaning, business attributes, and sentiment. And the dataset is from Yelp Kaggle competitions which can be found at: https://www.kaggle.com/yelp-dataset/yelp-dataset

Mining the text contains the preprocessing, training, testing and evaluating process. 

1. Preprocessing

In this processs, raw data needs to be cleaned and attributes needs to be determined. In this text dataset, many punctuations are considered to be cleaned.
Feature selection in text data is also very important. 

2. Training

This project imlements up to eight different classification models: Naive Bayes Classifier, Logistic Regression, Support Vector Machines, decision tress, random forests, bagged trees and Adaboost tree. 

3. Testing
The score function here used is zero-one-loss. Testing is realized by cross-validation.

4. Evaluation
Evaluation is based on testing result, and hypothesis testing can be used to test the performance between different models.

