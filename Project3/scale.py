import numpy as np # numpy is for vectors
import matplotlib.pyplot as plt
import csv

# Max/Min Group A
with open('Project3_data/test_data_4.csv', 'rt') as csvfile:
	train_data_1 = np.array(list(csv.reader(csvfile))).astype(np.float)

	# print('data before scaling')
	# print(train_data_1)
	#
	# plt.scatter(train_data_1[:,0], train_data_1[:,1], c='red', label = "Before scaling")
	# plt.show()

	X_std = np.copy(train_data_1)
	X_std[:,0] = (train_data_1[:,0] - train_data_1[:,0].mean()) / train_data_1[:,0].std()
	X_std[:,1] = (train_data_1[:,1] - train_data_1[:,1].mean()) / train_data_1[:,1].std()

	print('data after scaling')
	print(X_std)

	plt.scatter(X_std[:,0], X_std[:,1], c='red', label = "After scaling")
	plt.show()


with open('Project3_data/scaled/test_data_4.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for row in X_std:
		writer.writerow(row)
