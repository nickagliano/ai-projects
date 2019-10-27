import numpy as np # numpy is for vectors
import random
import math

# IMPORTANT NOTE:
#	The data set lists all male and then all female data points. Think about which
#	data points you should use for training and which for testing --
#	i.e. algorithm will fail if trained on one type of patters and tested on another

class Perceptron(object):

	def __init__(self, no_of_inputs, iterations=5000, learning_rate=2):
		# number of patterns? (2d array?)
		# desired output array? (is this supervised or unsupervised?)
		self.iterations = iterations # number of iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant'

		# initalize weights to random between (-.5, .5)
		#random.uniform(-0.5, 0.5)
		self.weights = [random.uniform(-0.5,0.5),random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)]
		print(self.weights)
		#self.weights = np.zeros(no_of_inputs + 1) # array of weights

	# unipolar hard activation function, called by train_hard function
	def predict_hard(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		if summation > 0:
		  activation = 1
		else: # since if condition is > 0, else means <= 0
		  activation = 0
		return activation

	# set error thresholds:
	# E < 10^-5 for Group A,
	# E < 10^2 for Group B,
	# E < 1.45 * 10^3 for Group C
	def train_hard(self, training_inputs, labels, stopping_criterion): # add parameter for '% of data used for training
		count = 0
		for _ in range(self.iterations):
			error = 0
			for inputs, label in zip(training_inputs, labels):
				prediction = self.predict_hard(inputs)
				update = self.learning_rate * (label - prediction)
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs # error = (label - prediction)
				self.weights[0] += self.learning_rate * (label - prediction)
				error += int(update != 0.0)
				#print(error)

			if error < stopping_criterion:
				print('Error:  ' + str(error))
				print('Iterations:  ' + str(count))
				return None
			count += 1


	# unipolar soft activation function
	def predict_soft(self, inputs):
		k = 3 # gain value
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0] # deos this need to be tweaked for soft activation function?
		activation = (1 / (1 + np.exp(k * -summation)))

		if (activation > .8):
			activation = 1
		elif activation < .2:
			activation = 0
		# print(activation)
		return activation


	def train_soft(self, training_inputs, labels, stopping_criterion):
		count = 0
		for _ in range(self.iterations):
			error = 0
			for inputs, label in zip(training_inputs, labels):
				prediction = self.predict_soft(inputs)
				update = self.learning_rate * (label - float(prediction))
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs # error = (label - prediction)
				self.weights[0] += self.learning_rate * (label - prediction)
				error += float(update != 0.0)

			if error < stopping_criterion:
				print('Error:  ' + str(error))
				print('Iterations:  ' + str(count))
				return None
			count += 1
			print(str(self.weights) + ' , error: ' + str(error) + ' , iterations: ' + str(count))

	# to easily present the findings!
	def print_results(self):
		print('weight of x: ' + str(self.weights[1]))
		print('weight of y: ' + str(self.weights[2]))
		print('weight of bias: ' + str(self.weights[0]))
