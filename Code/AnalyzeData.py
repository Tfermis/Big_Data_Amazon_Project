import pandas as pd
import os

class item:
	ID = -1
	PosCount = 0
	NegCount = 0
	Percentage = 0

	def __init__(self, ID):
            self.ID = ID
            self.PosCount = 0
            self.NegCount = 0

	def __hash__(self):
		return hash(self.ID)

	def __eq__(self, other):
		return self.ID == other.ID

	def __ne__(self, other):
		return not(self == other)


path = "/home/ee379k/EE379K/av23674/Big_Data_Amazon_Project"

attributePath = '/Attributes/'

originalFilesPath = '/Original Files/'


print "Opening DataSet"

Data = pd.read_csv(path + originalFilesPath + "train.csv")

ActionName =  Data.columns[0]


for x in xrange(1, len(Data.columns)):
	k = 0
	ListName = Data.columns[x]
	ActionList =  Data[Data.columns[0]]
	CurrentList =  Data[Data.columns[x]]

	CurrentDict = {}

	for i in xrange(CurrentList.size):
		if CurrentDict.has_key(CurrentList[i]):
			currentID = CurrentDict.get(CurrentList[i])
			if ActionList[i] == 1:
				currentID.PosCount += 1
			else:
				currentID.NegCount += 1
		else:
			CurrentDict[CurrentList[i]] = item(CurrentList[i])

	IDList = []
	PosList = []
	NegList = []

	for value in CurrentDict.values():
		IDList.append(value.ID)
		if value.PosCount > 0:
			PosList.append(value.PosCount / float(value.PosCount + value.NegCount))
		else:
			PosList.append(0)
		if value.NegCount > 0:
			NegList.append(value.NegCount / float(value.PosCount + value.NegCount))
		else:
			NegList.append(0)

	CurrentOut = pd.DataFrame({'ID' : IDList, 'Positive' : PosList, 'Negative': NegList})

	CurrentOut.to_csv(path + attributePath + ListName + ".csv")

	print ListName 


