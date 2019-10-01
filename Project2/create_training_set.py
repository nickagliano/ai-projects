import csv
import pprint
import random

dataset = []
male = []
female = []
training = []
testing = []
count = 0
with open('groupC.csv') as data:
    csvReader = csv.reader(data)
    for row in csvReader:
        dataset.append(row)


while count < 2000:
    male.append(dataset[count])
    count += 1

while count < 4000:
    female.append(dataset[count])
    count += 1

random.shuffle(male)
random.shuffle(female)

i = 0
while i < 2000:
    if i < 1500:
        training.append(male[i])
        training.append(female[i])
    else:
        testing.append(male[i])
        testing.append(female[i])
    i += 1

random.shuffle(training)
random.shuffle(testing)

with open('trainingC.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in training:
        writer.writerow(row)

with open('testingC.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for row in testing:
        writer.writerow(row)
