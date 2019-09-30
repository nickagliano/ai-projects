# import csv package for csv handling
import csv
# import numpy for vectors
import numpy as np

# import perceptron class
import Perceptron as Perceptron

print('starting main method')

print('...')

print('importing data for training')

# fake training data
training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0]) # here we store the expected outputs
									# making sure each index lines up with the index
									# of the input it's meant to represent

# import data, put in correct form for perceptron




print('...')

# -----------------------------------------------------------------------------
# hard, unipolar activation function, using 75% of data

print('training perceptron with hard, unipolar activation function')

hard_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs

print('training perceptron with hard, unipolar activation function, using 75% of data')

print('...')

perceptron.train_hard(training_inputs, labels)

print('perceptron done training, printing results...')

perceptron.print_results()

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
# perceptron.train_hard(training_inputs, labels)
#
# print('perceptron done training, printing results...')
#
# perceptron.print_results()
#
# print('...')
#
# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 75% of data
#
# soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs
#
# print('training perceptron with soft, unipolar activation function, using 75% of data')
#
# print('...')
#
# soft_ perceptron.train_soft(training_inputs, labels)
#
# print('perceptron done training, printing results...')
#
# soft_perceptron.print_results()
#
# print('...')
#
# # -----------------------------------------------------------------------------
# # soft, unipolar activation function, using 25% of data
#
# soft_perceptron = Perceptron.Perceptron(2) # 2 is number of inputs
#
# print('training perceptron with soft, unipolar activation function, using 25% of data')
#
# print('...')
#
# soft_perceptron.train_soft(training_inputs, labels)
#
# print('perceptron done training, printing results...')
#
# soft_perceptron.print_results()
#
# print('...')




print('finished')



# import groupA data
# with open('Project2_data/normalized/groupA.csv', 'rt') as csvfile:
# 	groupA = list(csv.reader(csvfile))
#
# 	print(groupA)
#

# main function for running perceptrons
# declare perceptrons,
# 	call perceptron methods with desired input parameters, datasets, etc.
