import numpy as np # numpy is for vectors
import random
import math
import csv

# import perceptron class
import Neuron as Neuron

# a_neuron = Neuron.Neuron(1) # neruon with 1 inputs (architecture a)
# with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
# 	training_1 = list(csv.reader(csvfile))
#
# time = np.array([])
# k_watts = np.array([])
# for row in training_1:
# 	time = np.append(time, float(row[0]))
# 	k_watts = np.append(k_watts, float(row[1]))
#
# a_neuron.train(time, k_watts)

# a_neuron = Neuron.Neuron(2) # neruon with 1 inputs (architecture a)
# with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
# 	training_1 = list(csv.reader(csvfile))
#
# time = np.array([])
# k_watts = np.array([])
# for row in training_1:
# 	time = np.append(time, float(row[0]))
# 	k_watts = np.append(k_watts, float(row[1]))
#
# a_neuron.train(time, k_watts)

a_neuron = Neuron.Neuron(3) # neruon with 1 inputs (architecture a)
with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
	training_1 = list(csv.reader(csvfile))

time = np.array([])
k_watts = np.array([])
for row in training_1:
	time = np.append(time, float(row[0]))
	k_watts = np.append(k_watts, float(row[1]))

a_neuron.train(time, k_watts)
