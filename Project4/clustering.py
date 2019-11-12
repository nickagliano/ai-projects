import pandas as pd
import math
import numpy as np
import random
from collections import defaultdict
from operator import add, truediv

#creates a dictionary with total instances of all words in all documents
def allWordCount(allWordDict,sentenceDict):
    for word in allWordDict:
        allWordDict[word] = int(allWordDict[word]) + int(sentenceDict[word])


def euclidean_distance(s1,s2):
    dist = 0
    for word1, word2 in zip(s1,s2):
        dist += (word1 + word2)**2
    # print(math.sqrt(dist))
    return math.sqrt(dist)

#pass in cluster which contains all sentences in that cluster so far
#
def updateweights(wk,cluster,sentence):
    # templist = [0]*len(sentence)
    i = 0
    for item in wk:
        wk[i] = item * (len(cluster))
        i += 1

    for s in cluster:
        wk = list(map(add,wk,s))



    wk = list(map(add,wk,sentence))
    # print("before")
    # print(templist)
    i = 0
    for item in wk:
        wk[i] = item / (len(cluster)+1)
        i += 1
    # print("after")
    # print(templist)
    # print(wk)
    # print("")
    return wk

def fcan(sentenceList):
    random.shuffle(sentenceList)
    mindist = 12
    centroids = []
    clusters = defaultdict(list)
    i = 1

    for sentence in sentenceList:
        dist = []
        newcentroid = True
        for centroid in centroids:
            dist.append(euclidean_distance(sentence,centroid))

        if len(dist) == 0:
            None

        elif min(dist) < mindist:
            # centroids[dist.index(min(dist))] = list(map(add,centroids[dist.index(min(dist))],updateweights(clusters[dist.index(min(dist))],sentence)))
            centroids[dist.index(min(dist))] = updateweights(centroids[dist.index(min(dist))],clusters[dist.index(min(dist))],sentence)
            clusters[dist.index(min(dist))].append(sentence)
            newcentroid = False

        if newcentroid:
            centroids.append(sentence)
            clusters[sentenceList.index(sentence)] = [sentence]

        i += 1

    return clusters


#read in csv and store into pandas dataframe
file = ('Project4_data/test_less_features.csv')
filedf = pd.read_csv(file)

sentenceList = []
sentenceindexlist = []
i = 0
while i < len(filedf):
    sentenceList.append(filedf.iloc[i].tolist())
    sentenceindexlist.append(filedf.iloc[i].tolist())
    i += 1

cluster = fcan(sentenceList)
count = 1
for a in cluster:
    print("Cluster: " + str(count))
    for b in cluster[a]:
        # print(b)
        print(sentenceList.index(b) + 1)
    count += 1
