import pandas as pd
import math
# import matplotlib.pyplot as plt
import numpy as np

#creates a dictionary with total instances of all words in all documents
def allWordCount(allWordDict,sentenceDict):
    for word in allWordDict:
        allWordDict[word] = int(allWordDict[word]) + int(sentenceDict[word])

#calculates term frequency for every term in a sentence
def tf(sentence):
    totalwords = len(sentence)
    termFrequency = {}
    for word in sentence:
        termFrequency[word] = sentence[word]/totalwords
    return termFrequency

#calculates inverse document frequency for all documents
def idf(idfDict,sentenceList):
    for sentence in sentenceList:
        for word in sentence:
            if sentence[word] > 0:
                idfDict[word] += 1

    for word in idfDict:
        idfDict[word] = math.log(len(sentenceList) / float(idfDict[word]))

    return idfDict

#calculates weights for every term in each sentence
def tf_idf(tfList,idfDict,tf_idfList):
    for sentence in tfList:
        tempList = []
        for word in sentence:
            tempList.append(sentence[word] * idfDict[word])

        tf_idfList.append(tempList)

    return tf_idfList

def euclidean_distance(s1,s2):
    dist = 0
    for word1, word2 in zip(s1,s2):
        dist += (word1 + word2)**2

    return math.sqrt(dist)

def kmeans(tf_idfList):
    #initialize centroids
    k = 3
    i = 1
    centroids = []
    while i < k+1:
        centroids.append(tf_idfList[i])
        i+=1

    # while i < 10:
    clusters = {}
    for n in range(k):
        clusters[n] = []

    num = 1
    for sentence in tf_idfList:
        dist = []
        for centroid in centroids:
            dist.append(euclidean_distance(sentence,centroid))
        topic = dist.index(min(dist))
        sentence.insert(0,num)
        clusters[topic].append(sentence)

        num+=1

    return clusters

#read in csv and store into pandas dataframe
file = ('Project4_data/test.csv')
filedf = pd.read_csv(file)

sentenceList = []

i = 0
while i < len(filedf):
    sentenceList.append(filedf.iloc[i].to_dict())
    i += 1

tfList = []
idfDict = {}
tf_idfList = []

allWordDict = {}
for row in filedf:
    allWordDict[row] = 0
    idfDict[row] = 0

for sentence in sentenceList:
    allWordCount(allWordDict,sentence)

for sentence in sentenceList:
    tfList.append(tf(sentence))

idfDict = idf(idfDict,sentenceList)
tempList = []
for a in filedf.iloc[0].items():
    tempList.append(a[0])

tf_idfList.append(tempList)

tf_idfList = tf_idf(tfList,idfDict,tf_idfList)

cluster = kmeans(tf_idfList[1:])
print(cluster[0])
