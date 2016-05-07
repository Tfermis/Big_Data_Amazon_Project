import os
import numpy as np
import pandas as pd


Files = ["Submit1.csv", "Submit2.csv"]
DataFrames = []

print "Reading in datasets"
for n in xrange(len(Files)):
	DataFrames.append(pd.read_csv(Files[n]))
	if(len(DataFrames[n]) != 58921):
		print Files[n] + " does not have the correct number of IDs"
		exit()

Values = {}
DataSetSize = len(DataFrames[1])

print "Creating dictionary"
temp = DataFrames[0]
for i in xrange(DataSetSize):
	Values[temp['id'][i]] = temp['ACTION'][i]

if(len(Values) != 58921):
	print "Dictionary is the wrong size"
	exit()

print "Picking best probability"
for j in xrange(len(Files)):
	currentFile = DataFrames[j]
	for k in xrange(DataSetSize):
		tempv1 = abs(Values[currentFile['id'][k]] - .5)
		tempv2 = abs(currentFile['ACTION'][k] - .5)
		if(tempv2 > tempv1):
			Values[currentFile['id'][k]] = currentFile['ACTION'][k]


actions = []

print "Preparing output"
for l in xrange(DataSetSize):
	prob  = Values[l + 1]
	actions.append(prob)

if(len(actions) != 58921):
	print "Dictionary is the wrong size"
	exit()

print "outputing"
content = ['id,ACTION']
for i in xrange(DataSetSize):
    content.append('%i,%f' %(i+1,actions[i]))
f = open('CombinedSubmission.csv', 'w')
f.write('\n'.join(content))
f.close()
