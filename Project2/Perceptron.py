import numpy as np # numpy is for matrices

# IMPORTANT NOTE:
#	The data set lists all male and then all female data points. Think about which
#	data points you should use for training and which for testing --
#	i.e. algorithm will fail if trained on one type of patters and tested on another

class Perceptron(object):

	def __init__(self, no_of_inputs, iterations=5000, learning_rate=0.1):
		# number of patterns? (2d array?)
		# desired output array? (is this supervised or unsupervised?)
		self.iterations = iterations # number of iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant'
		self.weights = np.zeros(no_of_inputs + 1) # array of weights

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
	# E < 10^-1 for Group B,
	# E < 5 * 10^-1 for Group C
	def train_hard(self, training_inputs, labels): # add parameter for '% of data used for training'
		for _ in range(self.iterations):
			for inputs, label in zip(training_inputs, labels):
				prediction = self.predict_hard(inputs)
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs # error = (label - prediction)
				self.weights[0] += self.learning_rate * (label - prediction)

	# set error thresholds:
	# E < 10^-5 for Group A,
	# E < 10^-1 for Group B,
	# E < 5 * 10^-1 for Group C
	# def train_soft(self, training_inputs, labels):
	# 	for _ in range(self.iterations):
	# 		for inputs, label in zip(training_inputs, labels):
	# 			prediction = self.predict(inputs)
	# 			self.weights[1:] += self.learning_rate * (label - prediction) * inputs # error = (label - prediction)
	# 			self.weights[0] += self.learning_rate * (label - prediction)

	# # unipolar soft activation function
	# def predict_soft(self, inputs):
	# 	summation = np.dot(inputs, self.weights[1:]) + self.weights[0] # deos this need to be tweaked for soft activation function?
	# 	if summation > condition: # figure out condition for soft activiation function
	# 	  activation = output # figure out how to calc output
	# 	else: #
	# 	  activation = output
	# 	return activation
