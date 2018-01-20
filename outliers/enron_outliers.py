#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)

for key in data_dict:
	if data_dict[key]["salary"] != 'NaN' and data_dict[key]["bonus"] != 'NaN':
		if data_dict[key]["salary"] > 1.0e6 and data_dict[key]["bonus"] > 5e6:
			print key

### your code below
print data.max()
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

############################
### LAY KENNETH L
### SKILLING JEFFREY K
### 8000000.0
############################
