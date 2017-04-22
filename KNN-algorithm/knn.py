#Author: Dipon Ghosh
#KNN algorithm implemented from scratch
#This is a simple machine learning classification algorithm. 
#This Algorithm uses EUCLEDIAN DISTANCE to get the neighbours. 
#Among k neighbours the maximum number of class label is 
#accepted as predicted class label. 

#NOTE: The following code is wtitten for IRIS-DATA only.

import csv
import math
import operator

def loadDataset(filename, inputset=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			inputset.append(dataset[x])


def eucledianDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow( (instance1[x] - instance2[x]), 2 )
	return math.sqrt(distance)

def getNeighbours(trainset, testinstance, k):
	distances = []
	length = len(testinstance) - 1
	for x in range(len(trainset)):
		dist = eucledianDistance(testinstance, trainset[x], length)
		distances.append( (trainset[x],dist) )
	distances.sort(key = operator.itemgetter(1))
	neighbour = []
	for x in range(k):
		neighbour.append(distances[x][0])
	return neighbour

def getResponse(neighbours):
	ClassVotes = {}
	for x in range(len(neighbours)):
		response = neighbours[x][-1]
		if response in ClassVotes:
			ClassVotes[response] += 1
		else:
			ClassVotes[response] = 1
	SortedVotes = sorted(ClassVotes.iteritems(), key = operator.itemgetter(1), reverse = True)
	return SortedVotes[0][0]

def getAccuracy(testset, prediction):
	correct = 0
	#print len(testset)
	for x in range(len(testset)):
		if testset[x][-1] == prediction[x]:
			correct = correct + 1
	return (correct/float(len(testset))) * 100.0


#prepare data
trainset=[]
testset=[]
loadDataset('iris-train.data',trainset)
loadDataset('iris-test.data',testset)
print 'Train Set: ' + repr(len(trainset))
print 'Test Set: ' + repr(len(testset))
#print trainset
#print testset

#generate predictions
prediction = []
k = 3 #we will consider 3 nearest neighbour
for x in range(len(testset)):
	neighbours = getNeighbours(trainset,testset[x],k)
	result = getResponse(neighbours)
	prediction.append(result)
	print('>predicted=' + repr(result) + ', actual= ' + repr(testset[x][-1]))
accuracy = getAccuracy(testset,prediction)
print('Accuracy: ' + repr(accuracy) + '%')

