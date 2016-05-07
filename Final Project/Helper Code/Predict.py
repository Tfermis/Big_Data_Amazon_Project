
import pandas as pd
import os


path = "/home/ee379k/EE379K/av23674/Big_Data_Amazon_Project"

attributePath = '/Attributes/'

originalFilesPath = '/Original Files/'

FeatureNames = []

Files = {}

def getProba(FeatureSet):
	for i in xrange(len(FeatureNames)):
		Data = pd.read_csv(path + originalFilesPath + "train.csv")
		Files[FeatureNames[i]] = 
		f = open(path + attributePath + FeatureNames[i] + '.csv', 'r')

	return 1




print "Opening Test DataSet"

Data = pd.read_csv(path + originalFilesPath + "test.csv")

NumFeatures = len(Data.columns)
DataSize = len(Data)

FeatureValues = []

for i in xrange(NumFeatures):
	FeatureNames.append(Data.columns[i])
	FeatureValues.append(Data[FeatureNames[i]])

content = ['id,ACTION']

for j in xrange(DataSize):
	currFeatures = []
	for k in xrange(len(FeatureNames)):
		currFeatures.append(FeatureValues[k][j])
	prob = getProba(currFeatures)
	content.append('%i,%f' %(j+1, prob))

f = open('BruteForceSubmission.csv', 'w')
f.write('\n'.join(content))
f.close()
