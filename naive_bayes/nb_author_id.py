#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction  time:", round(time()-t0, 3), "s"
t0 = time()
acc = accuracy_score(labels_test,pred)
print "accuracy calculating time:", round(time()-t0, 3), "s"
print(acc)

#########################################################
### OUTPUT ###
### no. of Chris training emails: 4406
### no. of Sara training emails: 4383
### training time: 0.681 s
### prediction  time: 0.91 s
### accuracy calculating time: 0.005 s
### 0.972693139151
#########################################################


