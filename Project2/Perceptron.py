import numpy as np

class Perceptron(object):

	def __init__(self, no_of_inputs, iterations=5000, learning_rate=0.1):
		# number of patterns? (2d array?)
		# desired output array? (is this supervised or unsupervised?)

		self.iterations = iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant'
		self.weights = np.zeros(no_of_inputs + 1) # array of weights

	# implement unipolar soft and unipolar hard activation function
	# (this is unipolar hard activation I'm 90% sure)
	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		if summation > 0:
		  activation = 1
		else: # <=
		  activation = 0
		return activation

	# set error thresholds:
	# E < 10^-5 for Group A,
	# E < 10^-1 for Group B,
	# E < 5 * 10^-1 for Group C
	def train(self, training_inputs, labels):
		for _ in range(self.iterations):
			for inputs, label in zip(training_inputs, labels):
				prediction = self.predict(inputs)
				self.weights[1:] += self.learning_rate * (label - prediction) * inputs # error = (label - prediction)
				self.weights[0] += self.learning_rate * (label - prediction)
