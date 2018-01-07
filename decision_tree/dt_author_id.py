#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
print "classifier initialization time:", round(time()-t0, 3), "s"

t0 = time()
clf = clf.fit(features_train,labels_train)
print "Fitting time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Predicting time:", round(time()-t0, 3), "s"

t0 = time()
acc = accuracy_score(labels_test,pred)
print "Accuracy time:", round(time()-t0, 3), "s"

print acc


#########################################################
### OUTPUT ###
### no. of Chris training emails: 7936
### no. of Sara training emails: 7884
### classifier initialization time: 0.002 s
### Fitting time: 5.69 s
### Predicting time: 0.005 s
### Accuracy time: 0.002 s
### 0.967007963595
#########################################################


