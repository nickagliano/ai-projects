# import csv package for csv handling
import csv
import random

# for plotting the data
import matplotlib.pyplot as plt

# import numpy for vectors
import numpy as np

# import perceptron class
import Perceptron as Perceptron

# -------------------------- UTILITY FUNCTIONS --------------------------------
# dataset is the full dataset
# percent training is what percent of the dataset will be used for training
def split_data(dataset, percent_training):
	# split datasets
	i = 0;
	training = []
	testing = []
	random.shuffle(dataset)
	n = 4000 * percent_training
	while i < 4000:
		if i < n:
			training.append(dataset[i])
		else:
			testing.append(dataset[i])
		i += 1
	return (training, testing)


def plot(perceptron, dataset, plot_title):
	red = ("red") # color for male data points (0)
	blue = ("blue") # color for female data points (1)
	maleX = []
	maleY = []
	femaleX = []
	femaleY = []
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	# line values corresponding to weights
	x = perceptron.weights[1]
	y = perceptron.weights[2]
	b = perceptron.weights[0]

	x_intercept = ((b * -1) / x)
	y_intercept = ((b * -1) / y)

	# plot decision line
	plt.plot([0, x_intercept], [y_intercept, 0])

	for el in dataset:
		# get data for calculating distance
		x0 = float(el[0])
		y0 = float(el[1])
		x1 = 0
		x2 = x_intercept
		y1 = y_intercept
		y2 = 0

		if int(el[2]) == 0: # if it's a male data point
			maleX.append(float(el[0]))
			maleY.append(float(el[1]))

			# test whether prediction was correct
			distance = ((x0 - x1)*(y2 - y1)) - ((y0-y1)*(x2-x1))
			if distance < 0:
				tn = tn + 1
			else:
				fp = fp + 1

		elif int(el[2]) == 1: # if it's a female data point'
			femaleX.append(float(el[0]))
			femaleY.append(float(el[1]))

			# test whether prediction was correct
			distance = ((x0 - x1)*(y2 - y1)) - ((y0-y1)*(x2-x1))
			if distance >= 0:
				tp = tp + 1
			else:
				fn = fn + 1

	print('tp=' + str(tp))
	print('fp=' + str(fp))
	print('tn=' + str(tn))
	print('fn=' + str(fn))

	# plotting the male points
	plt.scatter(maleX, maleY, c=red, label = "Males")

	# plotting the female points
	plt.scatter(femaleX, femaleY, c=blue, label = "Females")

	# giving a title to my graph
	plt.title(plot_title)
	plt.xlabel('Height (in feet)')
	plt.ylabel('Weight (in pounds)')

	# show a legend on the plot
	plt.legend()

	# function to show the plot
	plt.show()



# -----------------------------------------------------------------------------
# populate groupA list
# with open('Project2_data/normalized/groupA.csv', 'rt') as csvfile:
# 	groupA = list(csv.reader(csvfile))

# -----------------------------------------------------------------------------
# hard, unipolar activation function, using 75% of data from group A
#
with open('Project2_data/training_testing/training_A_75.csv', 'rt') as csvfile:
	training_A_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_A_25.csv', 'rt') as csvfile:
	testing_A_25 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_A_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 0.00001)

hard_perceptron.print_results()


plot(hard_perceptron, training_A_75, 'Group A Training, 75%')
plot(hard_perceptron, testing_A_25, 'Group A Testing, 75%')

# # # -----------------------------------------------------------------------------
# # hard, unipolar activation function, using 25% of data from group A
#
with open('Project2_data/training_testing/training_A_25.csv', 'rt') as csvfile:
	training_A_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_A_75.csv', 'rt') as csvfile:
	testing_A_75 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.25)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_A_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 0.00001)

hard_perceptron.print_results()

plot(hard_perceptron, training_A_25, 'Group A Training, 25%')
plot(hard_perceptron, testing_A_75, 'Group A Testing, 25%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 75% of data from group C

with open('Project2_data/training_testing/training_A_75.csv', 'rt') as csvfile:
	training_A_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_A_25.csv', 'rt') as csvfile:
	testing_A_25 = list(csv.reader(csvfile))

soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# datasets = split_data(groupB, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_A_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, .00001)

soft_perceptron.print_results()

plot(soft_perceptron, training_A_75, 'Group A Training, 75%')
plot(soft_perceptron, testing_A_25, 'Group A Testing, 75%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 25% of data from group A

with open('Project2_data/training_testing/training_A_25.csv', 'rt') as csvfile:
	training_A_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_A_75.csv', 'rt') as csvfile:
	testing_A_75 = list(csv.reader(csvfile))

soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# datasets = split_data(groupB, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_A_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, .00001)

soft_perceptron.print_results()

plot(soft_perceptron, training_A_25, 'Group A Training, 25%')
plot(soft_perceptron, testing_A_75, 'Group A Testing, 25%')



# -----------------------------------------------------------------------------
# print('---------------------- GROUP B ---------------------------------------')
#
# # populate groupB list
# with open('Project2_data/training_testing/training_B_75.csv', 'rt') as csvfile:
# 	groupB = list(csv.reader(csvfile))
#
with open('Project2_data/training_testing/training_B_75.csv', 'rt') as csvfile:
	training_B_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_B_25.csv', 'rt') as csvfile:
	testing_B_25 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_B_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 100)

hard_perceptron.print_results()


plot(hard_perceptron, training_B_75, 'Group B Training, 75%')
plot(hard_perceptron, testing_B_25, 'Group B Testing, 75%')
#
# # # -----------------------------------------------------------------------------
# # hard, unipolar activation function, using 25% of data from group A
#
with open('Project2_data/training_testing/training_B_25.csv', 'rt') as csvfile:
	training_B_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_B_75.csv', 'rt') as csvfile:
	testing_B_75 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.25)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_B_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 100)

hard_perceptron.print_results()

plot(hard_perceptron, training_B_25, 'Group B Training, 25%')
plot(hard_perceptron, testing_B_75, 'Group B Testing, 25%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 75% of data from group B

with open('Project2_data/training_testing/training_B_75.csv', 'rt') as csvfile:
	training_B_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_B_25.csv', 'rt') as csvfile:
	testing_B_25 = list(csv.reader(csvfile))


soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# datasets = split_data(groupB, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_B_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, 100)

soft_perceptron.print_results()

plot(soft_perceptron, training_B_75, 'Group B Training, 75%')
plot(soft_perceptron, testing_B_25, 'Group B Testing, 75%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 25% of data from group B

with open('Project2_data/training_testing/training_B_25.csv', 'rt') as csvfile:
	training_B_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_B_75.csv', 'rt') as csvfile:
	testing_B_75 = list(csv.reader(csvfile))

soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# datasets = split_data(groupB, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_B_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, 100)

soft_perceptron.print_results()

plot(soft_perceptron, training_B_25, 'Group B Training, 25%')
plot(soft_perceptron, testing_B_75, 'Group B Testing, 25%')
# # -----------------------------------------------------------------------------
# print('---------------------- GROUP C ---------------------------------------')
#
# # populate groupC list
# with open('Project2_data/normalized/groupC.csv', 'rt') as csvfile:
# 	groupC = list(csv.reader(csvfile))
#
# # -----------------------------------------------------------------------------
# # hard, unipolar activation function, using 75% of data from group C
#
with open('Project2_data/training_testing/training_C_75.csv', 'rt') as csvfile:
	training_C_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_C_25.csv', 'rt') as csvfile:
	testing_C_25 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_C_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 1450)

hard_perceptron.print_results()


plot(hard_perceptron, training_C_75, 'Group C Training, 75%')
plot(hard_perceptron, testing_C_25, 'Group C Testing, 75%')

# # -----------------------------------------------------------------------------
# hard, unipolar activation function, using 25% of data from group A

with open('Project2_data/training_testing/training_C_25.csv', 'rt') as csvfile:
	training_C_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_C_75.csv', 'rt') as csvfile:
	testing_C_75 = list(csv.reader(csvfile))

# declare and instantiate perceptron
hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# split data into training and testing datasets
# datasets = split_data(groupA, 0.25)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_C_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

hard_perceptron.train_hard(training_data_no_labels, labels, 1450)

hard_perceptron.print_results()

plot(hard_perceptron, training_C_25, 'Group C Training, 25%')
plot(hard_perceptron, testing_C_75, 'Group C Testing, 25%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 75% of data from group C
#
with open('Project2_data/training_testing/training_C_75.csv', 'rt') as csvfile:
	training_C_75 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_C_25.csv', 'rt') as csvfile:
	testing_C_25 = list(csv.reader(csvfile))

soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

# datasets = split_data(groupB, 0.75)
# training_data = datasets[0]
# testing_data = datasets[1]

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_C_75:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, 1450)

soft_perceptron.print_results()

plot(soft_perceptron, training_C_75, 'Group C Training, 75%')
plot(soft_perceptron, testing_C_25, 'Group C Testing, 75%')

# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 25% of data from group C

with open('Project2_data/training_testing/training_C_25.csv', 'rt') as csvfile:
	training_C_25 = list(csv.reader(csvfile))
with open('Project2_data/training_testing/testing_C_75.csv', 'rt') as csvfile:
	testing_C_75 = list(csv.reader(csvfile))

soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

training_data_no_labels = []
labels = np.array([])

# iterate through training data to take off labels and put in the form for perceptron class
for row in training_C_25:
	training_data_no_labels.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
	labels = np.append(labels, int(row[2])) # append class for that row into labels vector

soft_perceptron.train_soft(training_data_no_labels, labels, 1450)

soft_perceptron.print_results()

plot(soft_perceptron, training_C_25, 'Group C Training, 25%')
plot(soft_perceptron, testing_C_75, 'Group C Testing, 25%')
# # -----------------------------------------------------------------------------



print('finished')
