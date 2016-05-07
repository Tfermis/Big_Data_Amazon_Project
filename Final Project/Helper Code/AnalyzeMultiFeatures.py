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
ActionList =  Data[ActionName]


for x in xrange(1, len(Data.columns)):
	temp = x + 1
	for y in xrange(temp, len(Data.columns)):
		k = 0
		ListName = Data.columns[x]
		CurrentList =  Data[Data.columns[x]]
		ListName2 = Data.columns[y]
		CurrentList2 =  Data[Data.columns[y]]
		CurrentDict = {}
		for i in xrange(CurrentList.size):
			ID = str(CurrentList[i]) + ":" + str(CurrentList2[i])
			if CurrentDict.has_key(ID):
				currentID = CurrentDict.get(ID)
				if ActionList[i] == 1:
					currentID.PosCount += 1
				else:
					currentID.NegCount += 1
			else:
				CurrentDict[ID] = item(ID)
				if ActionList[i] == 1:
					CurrentDict[ID].PosCount += 1
				else:
					CurrentDict[ID].NegCount += 1
		PosList = []
		NegList = []
		NumOccurences = []
		for value in CurrentDict.values():
			if value.PosCount > 0:
				PosList.append(value.PosCount / float(value.PosCount + value.NegCount))
			else:
				PosList.append(0)
			if value.NegCount > 0:
				NegList.append(value.NegCount / float(value.PosCount + value.NegCount))
			else:
				NegList.append(0)
			NumOccurences.append(value.NegCount + value.PosCount)

		print ListName + ":" + ListName2 + "||" + str(sum(PosList)/float(len(NumOccurences)))
# -------------------------


