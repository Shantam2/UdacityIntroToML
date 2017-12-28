#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


clf = SVC(kernel="rbf",C=10000)
t0=time()
###features_train = features_train[:len(features_train)/100] 
###labels_train = labels_train[:len(labels_train)/100] 
clf.fit(features_train,labels_train)
print "Training time:", round(time()-t0, 3), "s"
t0=time()
pred = clf.predict(features_test)
print "Predicting time:", round(time()-t0, 3), "s"
acc = accuracy_score(labels_test,pred)
print(acc)

ans1=pred[10]
ans2=pred[26]
ans3=pred[50]

print(ans1,ans2,ans3)   

#########################################################
### OUTPUT ###
### no. of Chris training emails: 6161
### no. of Sara training emails: 6143
### Training time: 120.198 s
### Predicting time: 50.295 s
### >>> print(acc)
### 0.982555934774
### 
### >>> from collections import Counter
### >>> Counter(pred)
### Counter({0: 881, 1: 877})
#########################################################


