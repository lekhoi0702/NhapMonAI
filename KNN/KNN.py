import csv
import numpy as np
import math
import random
from collections import Counter
import pandas as pd

dataSet = pd.read_csv('Iris.csv')
dataSet.isna().sum()
dataSet.duplicated().sum()
print(dataSet)
dataSet = np.delete(dataSet,0,0) #Xoá header
dataSet = np.delete(dataSet,0,1) #Xoá id
random.shuffle(dataSet)

num_rows = dataSet.shape[0]
num_train = int(0.8 * num_rows) 

dataTrain = dataSet[:num_train] #80 % train
dataTest = dataSet[num_train:] # 20% test

#Tính khoản cách giữa 2 điểm
def euclidean_distance(x1,x2):
    result = 0
    for i in range(4):
        result += (float(x1[i]) - float(x2[i]))**2
    return math.sqrt(result)


#Chọn ra k điểm gần nhất với item
def get_neighbors(dataTrain,item,k):
    distances = []
    for i,value in enumerate(dataTrain):
        distance = euclidean_distance(item,value)
        distances.append((i,distance))
    distances.sort(key = lambda x:x[1])
    neighbors = [i for i, _ in distances[:k]]
    result = [dataTrain[i][-1] for i in  neighbors]
    return result

#Chọn ra nhãn xuất hiện nhiều nhất
def vote(neighbors,k):
    neighbors = get_neighbors(dataTrain, item, k)
    neighbor = Counter(neighbors).most_common()
    return neighbor[0]
    
#main
k = 3
numOfRightAnwser = 0
for item in dataTest:
    knn = get_neighbors(dataTrain, item, k)
    result = vote(knn, k)
    numOfRightAnwser += item[-1] == result[0]
    print("name:{} --> predict:{}".format(item[-1],result[0]))
print("Accuracy", numOfRightAnwser/len(dataTest))
    


    








