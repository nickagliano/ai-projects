import csv

# Max/Min Group A
with open('Project2_data/raw/groupA.csv', 'rb') as csvfile:
	groupA = list(csv.reader(csvfile))

maxHeightA = 0
minHeightA = 99999
maxWeightA = 0
minWeightA = 99999

for el in groupA:
	if float(el[0]) > float(maxHeightA):
		maxHeightA = el[0]
	if float(el[0]) < float(minHeightA):
		minHeightA = el[0]
	if float(el[1]) > float(maxWeightA):
		maxWeightA = el[1]
	if float(el[1]) < float(minWeightA):
		minWeightA = el[1]

print('GROUP A')
print('maxHeightA=' + str(maxHeightA))
print('minHeightA=' + str(minHeightA))
print('maxWeightA=' + str(maxWeightA))
print('minWeightA=' + str(minWeightA))
print('**********************************')

# Normalize Group A
with open('Project2_data/normalized/groupA.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for el in groupA:
		normalizedHeight = (((float(el[0]) - float(minHeightA)) / (float(maxHeightA) - float(minHeightA))))
		normalizedWeight = (((float(el[1]) - float(minWeightA)) / (float(maxWeightA) - float(minWeightA))))
		writer.writerow([normalizedHeight, normalizedWeight, el[2]])


# Find Max/Min Group B
with open('Project2_data/raw/groupB.csv', 'rb') as csvfile:
	groupB = list(csv.reader(csvfile))

maxHeightB = 0
minHeightB = 99999
maxWeightB = 0
minWeightB = 99999

for el in groupB:
	if float(el[0]) > float(maxHeightB):
		maxHeightB = el[0]

	if float(el[0]) < float(minHeightB):
		minHeightB = el[0]

	if float(el[1]) > float(maxWeightB):
		maxWeightB = el[1]

	if float(el[1]) < float(minWeightB):
		minWeightB = el[1]

print('GROUP B')
print('maxHeightB=' + str(maxHeightB))
print('minHeightB=' + str(minHeightB))
print('maxWeightB=' + str(maxWeightB))
print('minWeightB=' + str(minWeightB))
print('**********************************')


# Normalize Group B
with open('Project2_data/normalized/groupB.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for el in groupB:
		normalizedHeight = ((float(el[0]) - float(minHeightB)) / (float(maxHeightB) - float(minHeightB)))
		normalizedWeight = ((float(el[1]) - float(minWeightB)) / (float(maxWeightB) - float(minWeightB)))
		writer.writerow([normalizedHeight, normalizedWeight, el[2]])


# Max/Min Group C
with open('Project2_data/raw/groupC.csv', 'rb') as csvfile:
	groupC = list(csv.reader(csvfile))

maxHeightC = 0
minHeightC = 99999
maxWeightC = 0
minWeightC = 99999

for el in groupC:
	if float(el[0]) > float(maxHeightC):
		maxHeightC = el[0]

	if float(el[0]) < float(minHeightC):
		minHeightC = el[0]

	if float(el[1]) > float(maxWeightC):
		maxWeightC = el[1]

	if float(el[1]) < float(minWeightC):
		minWeightC = el[1]

print('GROUP C')
print('maxHeightC=' + str(maxHeightC))
print('minHeightC=' + str(minHeightC))
print('maxWeightC=' + str(maxWeightC))
print('minWeightC=' + str(minWeightC))

# Normalize Group C
with open('Project2_data/normalized/groupC.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for el in groupC:
		normalizedHeight = ((float(el[0]) - float(minHeightC)) / (float(maxHeightC) - float(minHeightC)))
		normalizedWeight = ((float(el[1]) - float(minWeightC)) / (float(maxWeightC) - float(minWeightC)))
		writer.writerow([normalizedHeight, normalizedWeight, el[2]])
