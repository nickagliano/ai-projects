import numpy as np # numpy is for vectors
import random
import math

class Neuron(object):

	def __init__(self, no_of_inputs, iterations=5000, learning_rate=1):
		self.iterations = iterations # number of iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant'
		self.weights = [0] * (no_of_inputs + 1) # declare weight vector
		for i, w in enumerate(self.weights): # initialize weight vector
			self.weights[i] = random.uniform(0, 10)
		print(self.weights)
