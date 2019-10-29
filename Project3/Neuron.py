import numpy as np # numpy is for vectors
import random
import math
import matplotlib.pyplot as plt

class Neuron(object):

	def __init__(self, no_of_inputs, iterations=1000, learning_rate=.001):
		self.iterations = iterations # number of iterations
		self.learning_rate = learning_rate # AKA 'alpha', 'learning constant', 'eta'
		self.weights = [0] * (no_of_inputs + 1) # declare weight vector size of inputs + 1
		self.total_error_array = []
		for i, w in enumerate(self.weights): # initialize weight vector
			self.weights[i] = random.uniform(-1, 1)
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


	def update_weights(self, time, k_watts, error_array):
		predictions = np.array([])

		for x in time:
			predictions = np.append(predictions, self.predict(x))

		if (len(self.weights)) == 2:
			self.weights[1] -= 2 * self.learning_rate * time.T.dot(error_array)
			self.weights[0] -= self.learning_rate * error_array.sum()
		if (len(self.weights)) == 3:
			self.weights[2] -= 2 * self.learning_rate * (time**2).T.dot(error_array)
			self.weights[1] -= 2 * self.learning_rate * time.T.dot(error_array)
			self.weights[0] -= self.learning_rate * error_array.sum()
		if (len(self.weights)) == 4:
			self.weights[3] -= 2 * self.learning_rate * (time**3).T.dot(error_array)
			self.weights[2] -= 2 * self.learning_rate * (time**2).T.dot(error_array)
			self.weights[1] -= 2 * self.learning_rate * time.T.dot(error_array)
			self.weights[0] -= self.learning_rate * error_array.sum()


	def get_error(self, time, k_watts):
		error_array = np.array([])
		for (x, y) in zip(time, k_watts):
			prediction = self.predict(x)
			error_array = np.append(error_array, prediction - y)
		self.total_error_array.append((error_array**2).sum() / 2)
		return error_array


	def train(self, time, k_watts, plot_title):
		count = 0
		self.total_error_array = []
		for i in range(self.iterations):
			error_array = self.get_error(time, k_watts)
			self.update_weights(time, k_watts, error_array)

			count += 1
			if (count % 1000 == 0):
				self.plot(time, k_watts, plot_title)
		print('Sum-squared error for ' + str(plot_title) + ': ' + str(self.total_error_array[-1]))
		self.plot_error()

	def plot(self, time, k_watts, plot_title):
		x_vector = np.linspace(-2, 2, 100)
		y_vector = []

		# populate y_vector
		if len(self.weights) == 4: # if cubic
			for x in x_vector:
				y_vector.append(self.weights[3]*(x**3) + self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 3: # if quadratic
			for x in x_vector:
				y_vector.append(self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 2: # if linear
			for x in x_vector:
				y_vector.append(self.weights[1]*x + self.weights[0])


		plt.plot(x_vector, y_vector, '-b', label='Predicted')
		plt.scatter(time, k_watts, c='red', label = "Actual")
		plt.title(plot_title)
		plt.text(-1.5, 4, "Iterations: " + str(self.iterations) + ', alpha = ' + str(self.learning_rate))
		plt.xlabel('Time (in hours)')
		plt.ylabel('Energy (in kW)')
		plt.legend()
		plt.show()

	def plot_error(self):
		x_vector = np.linspace(0, self.iterations, self.iterations)
		y_vector = self.total_error_array

		plt.plot(x_vector, y_vector, '-r', label='Error trend')
		# plt.scatter(x_vector, y_vector, c='red', label = "Error point")
		plt.title('Sum-squared Error Over Iterations of LMS Algorithm')
		plt.xlabel('Iteration number')
		plt.ylabel('Sum-squared error')
		plt.legend()
		plt.show()

	def test(self, test_times, test_k_watts, plot_title):
		x_vector = test_times.tolist()
		y_vector_test = test_k_watts.tolist()
		y_vector_predict = []

		final_total_error = 0

		# populate y_vector
		if len(self.weights) == 4: # if cubic
			for x in x_vector:
				y_vector_predict.append(self.weights[3]*(x**3) + self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 3: # if quadratic
			for x in x_vector:
				y_vector_predict.append(self.weights[2]*(x**2) + self.weights[1]*x + self.weights[0])
		elif len(self.weights) == 2: # if linear
			for x in x_vector:
				y_vector_predict.append(self.weights[1]*x + self.weights[0])

		for (p, t) in zip(y_vector_predict, y_vector_test):
			final_total_error += (p - t)**2

		print('Sum-squared error for ' + str(plot_title) + ': ' + str(final_total_error))

		plt.plot(x_vector, y_vector_predict, '-b', label='Predicted')
		plt.scatter(test_times, y_vector_test, c='red', label = "Actual")
		plt.text(-1.5, 2, "Total Sum-squared Error: " + str(final_total_error))
		plt.title(plot_title)
		plt.xlabel('Time (in hours)')
		plt.ylabel('Energy (in kW)')
		plt.legend()
		plt.show()


	def print_results(self):
		i = 0
		for w in self.weights:
			print('weight of x' + str(i) + ': ' + str(self.weights[i]))
			i += 1
