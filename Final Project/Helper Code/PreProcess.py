


import pandas as pd

path = "/home/ee379k/EE379K/av23674/Big_Data_Amazon_Project"

attributePath = '/Attributes/'

originalFilesPath = '/Original Files/'

print "Opening Test DataSet"

Data = pd.read_csv(path + originalFilesPath + "train.csv")

DataSetSize = len(Data)

colNames = []
colValues = []

for i in xrange(len(Data.columns)):
	colNames.append(Data.columns[i])
	colValues.append(Data[Data.columns[i]])

content = []
for j in xrange(DataSetSize):
	line = str(colValues[0][j])
	for k in xrange(1,len(colValues)):
		line = line + " " + str(k) + ":" + str(colValues[k][j])
	content.append(line)

f = open('train.txt', 'w')
f.write('\n'.join(content))
f.close()
