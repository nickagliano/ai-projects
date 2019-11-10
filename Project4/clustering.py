import pandas as pd
import math

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

#read in csv and store into pandas dataframe
file = ('test.csv')
filedf = pd.read_csv(file)

sentenceList = []

i = 0
while i < len(filedf):
    sentenceList.append(filedf.iloc[i].to_dict())
    i += 1

# sentenceDict1 = filedf.iloc[0].to_dict()
# sentenceDict2 = filedf.iloc[1].to_dict()
# sentenceDict3 = filedf.iloc[2].to_dict()
# sentenceDict4 = filedf.iloc[3].to_dict()
#
# sentenceList.append(sentenceDict1)
# sentenceList.append(sentenceDict2)
# sentenceList.append(sentenceDict3)
# sentenceList.append(sentenceDict4)

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

# print(tfList)
# print(idfDict)
for a in tf_idfList:
    print(a)
