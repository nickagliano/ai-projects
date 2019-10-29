import numpy as np # numpy is for vectors
import random
import math
import csv

# import perceptron class
import Neuron as Neuron

a_neuron = Neuron.Neuron(1) # neruon with 1 inputs (architecture a)

with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
	training_1 = list(csv.reader(csvfile))

time1 = np.array([])
k_watts1 = np.array([])
for row in training_1:
	time1 = np.append(time1, float(row[0]))
	k_watts1 = np.append(k_watts1, float(row[1]))
a_neuron.train(time1, k_watts1, 'Architecture A Trained on train_data_1')

with open('Project3_data/scaled/train_data_2.csv', 'rt') as csvfile:
	training_2 = list(csv.reader(csvfile))

time2 = np.array([])
k_watts2 = np.array([])
for row in training_2:
	time2 = np.append(time2, float(row[0]))
	k_watts2 = np.append(k_watts2, float(row[1]))
a_neuron.train(time2, k_watts2, 'Architecture A Trained on train_data_2')

with open('Project3_data/scaled/train_data_3.csv', 'rt') as csvfile:
	training_3 = list(csv.reader(csvfile))

time3 = np.array([])
k_watts3 = np.array([])
for row in training_3:
	time3 = np.append(time3, float(row[0]))
	k_watts3 = np.append(k_watts3, float(row[1]))
a_neuron.train(time3, k_watts3, 'Architecture A Trained on train_data_3')

with open('Project3_data/scaled/test_data_4.csv', 'rt') as csvfile:
	test_4 = list(csv.reader(csvfile))

time4 = np.array([])
k_watts4 = np.array([])
for row in test_4:
	time4 = np.append(time4, float(row[0]))
	k_watts4 = np.append(k_watts4, float(row[1]))
a_neuron.test(time4, k_watts4, 'Testing Architecture A')


# b_neuron = Neuron.Neuron(2) # neruon with 3 inputs (architecture c)
#
# with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
# 	training_1 = list(csv.reader(csvfile))
#
# time1 = np.array([])
# k_watts1 = np.array([])
# for row in training_1:
# 	time1 = np.append(time1, float(row[0]))
# 	k_watts1 = np.append(k_watts1, float(row[1]))
# b_neuron.train(time1, k_watts1, 'Architecture B Trained on train_data_1')
#
# with open('Project3_data/scaled/train_data_2.csv', 'rt') as csvfile:
# 	training_2 = list(csv.reader(csvfile))
#
# time2 = np.array([])
# k_watts2 = np.array([])
# for row in training_2:
# 	time2 = np.append(time2, float(row[0]))
# 	k_watts2 = np.append(k_watts2, float(row[1]))
# b_neuron.train(time2, k_watts2, 'Architecture B Trained on train_data_2')
#
# with open('Project3_data/scaled/train_data_3.csv', 'rt') as csvfile:
# 	training_3 = list(csv.reader(csvfile))
#
# time3 = np.array([])
# k_watts3 = np.array([])
# for row in training_3:
# 	time3 = np.append(time3, float(row[0]))
# 	k_watts3 = np.append(k_watts3, float(row[1]))
# b_neuron.train(time3, k_watts3, 'Architecture B Trained on train_data_3')
#
# with open('Project3_data/scaled/test_data_4.csv', 'rt') as csvfile:
# 	test_4 = list(csv.reader(csvfile))
#
# time4 = np.array([])
# k_watts4 = np.array([])
# for row in test_4:
# 	time4 = np.append(time4, float(row[0]))
# 	k_watts4 = np.append(k_watts4, float(row[1]))
# b_neuron.test(time4, k_watts4, 'Testing Architecture B')

# c_neuron = Neuron.Neuron(3) # neruon with 3 inputs (architecture c)
#
# with open('Project3_data/scaled/train_data_1.csv', 'rt') as csvfile:
# 	training_1 = list(csv.reader(csvfile))
#
# time1 = np.array([])
# k_watts1 = np.array([])
# for row in training_1:
# 	time1 = np.append(time1, float(row[0]))
# 	k_watts1 = np.append(k_watts1, float(row[1]))
# c_neuron.train(time1, k_watts1, 'Architecture C Trained on train_data_1')
#
# with open('Project3_data/scaled/train_data_2.csv', 'rt') as csvfile:
# 	training_2 = list(csv.reader(csvfile))
#
# time2 = np.array([])
# k_watts2 = np.array([])
# for row in training_2:
# 	time2 = np.append(time2, float(row[0]))
# 	k_watts2 = np.append(k_watts2, float(row[1]))
# c_neuron.train(time2, k_watts2, 'Architecture C Trained on train_data_2')
#
# with open('Project3_data/scaled/train_data_3.csv', 'rt') as csvfile:
# 	training_3 = list(csv.reader(csvfile))
#
# time3 = np.array([])
# k_watts3 = np.array([])
# for row in training_3:
# 	time3 = np.append(time3, float(row[0]))
# 	k_watts3 = np.append(k_watts3, float(row[1]))
# c_neuron.train(time3, k_watts3, 'Architecture C Trained on train_data_3')
#
# with open('Project3_data/scaled/test_data_4.csv', 'rt') as csvfile:
# 	test_4 = list(csv.reader(csvfile))
#
# time4 = np.array([])
# k_watts4 = np.array([])
# for row in test_4:
# 	time4 = np.append(time4, float(row[0]))
# 	k_watts4 = np.append(k_watts4, float(row[1]))
# c_neuron.test(time4, k_watts4, 'Testing Architecture C')
