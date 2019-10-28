import numpy as np # numpy is for vectors
import random
import math
import matplotlib.pyplot as plt

class Neuron(object):

	def __init__(self, no_of_inputs, iterations=1000, learning_rate=0.0001):
		self.iterations = iterations # number of iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant'
		self.weights = [0] * (no_of_inputs + 1) # declare weight vector
		for i, w in enumerate(self.weights): # initialize weight vector
			self.weights[i] = random.uniform(0, 2)
		print('initial weights: ')
		print(self.weights)

	def predict(self, x):
		w = self.weights
		prediction = 0
		i = 0
		for wi in w:
			prediction += wi * (x**i)
			i += 1
		return prediction


	def error(self, time, k_watts):
		error = 0
		for (x, y) in zip(time, k_watts):
			error += (self.predict(x) - y)**2
		return error


	def derive(self, time, k_watts):
		d_array = [0] * len(self.weights)

		for (x, y) in zip(time, k_watts):
			i = 0
			for w in self.weights:
				if i == 3:
					d_array[i] = -2 * (y - self.predict(x)) * (x**3)
				elif i == 2:
					d_array[i] = -2 * (y - self.predict(x)) * (x**2)
				elif i == 1:
					d_array[i] = -2 * (y - self.predict(x)) * (x)
				else:
					d_array[i] = -2 * (y - self.predict(x))
				i+=1

		i = 0
		for d in d_array:
			d_array[i] /= len(time)
			i += 1

		return d_array

	def train(self, time, k_watts):
		count = 0
		for _ in range(self.iterations):
			self.update_weights(time, k_watts)
			print(self.error(time, k_watts))
			count += 1
			if (count % 100 == 0):
				self.plot(time, k_watts, ':)')

	def update_weights(self, time, k_watts):
		d_array = self.derive(time, k_watts)
		print('weights')
		self.print_results()
		# print('d array:')
		# print(d_array)
		i = 0
		for d in d_array:
			self.weights[i] = self.weights[i] - (self.learning_rate * d)
			i += 1

	def print_results(self):
		i = 0
		for w in self.weights:
			print('weight of x' + str(i) + ': ' + str(self.weights[i]))
			i += 1

	def plot(self, time, k_watts, plot_title):
		# line values corresponding to weights (need to adjust for polynomials)

		x_vector = np.linspace(5, 20, 100)
		y_vector = []

		if len(self.weights) == 4:
			for x in x_vector:
				y_vector.append(self.weights[3]*(x**3) + self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 3:
			for x in x_vector:
				y_vector.append(self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 2:
			for x in x_vector:
				y_vector.append(self.weights[1]*x + self.weights[0])


		plt.plot(x_vector, y_vector, '-b', label='Predicted')
		plt.scatter(time, k_watts, c='red', label = "Actual")
		# plt.ylim([0,12])
		plt.title(plot_title)
		plt.xlabel('Time (in hours)')
		plt.ylabel('Energy (in kW)')
		plt.legend()
		plt.show()
