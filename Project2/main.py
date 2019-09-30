# import csv package for csv handling
import csv

# import numpy for vectors
import numpy as np

# import perceptron class
import Perceptron as Perceptron

print('starting main method')

print('...')

print('importing data for training')

# declare full dataset, will be populated by csvfile reader
full_dataset = []

# declare empty labels array, will hold the class of the data
labels = np.array([])

# populate full_dataset array
with open('Project2_data/normalized/groupA.csv', 'rt') as csvfile:
	groupA = list(csv.reader(csvfile))

	for row in groupA:
		full_dataset.append(np.array([float(row[0]), float(row[1])])) # append to 2d array with x, y values
		labels = np.append(labels, int(row[2])) # append class for that row into labels vector

print('...')

# -----------------------------------------------------------------------------
# hard, unipolar activation function, using 75% of data

hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

print('training perceptron with hard, unipolar activation function, using 75% of data')

print('...')

datasets = split_data(full_dataset, 75)
training_data = datasets[0]
testing_data = datasets[1]

hard_perceptron.train_hard(training_data, labels)

print('perceptron done training, printing results...')

hard_perceptron.print_results()

print('...')

# # -----------------------------------------------------------------------------
# # hard, unipolar activation function, using 25% of data
#
# hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs
#
# print('training perceptron with hard, unipolar activation function, using 25% of data')
#
# print('...')
#
# hard_perceptron.train_hard(full_dataset, labels)
#
# print('perceptron done training, printing results...')
#
# hard_perceptron.print_results()
#
# print('...')
#
# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 75% of data
#

print('finished')



# dataset is the full dataset
# percent training is what percent of the dataset will be used for training
def split_data(dataset, percent_training):
	# split datasets

	# using pop() and random number in a range

	# training = percent_of full dataset used for training

	# testing = full dataset - percent used for training

	return (training, testing)
