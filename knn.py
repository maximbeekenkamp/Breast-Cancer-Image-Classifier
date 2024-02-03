from supervisedlearner import SupervisedLearner
import numpy as np

class KNNClassifier(SupervisedLearner):
	def __init__(self, feature_funcs, k):
		super(KNNClassifier, self).__init__(feature_funcs)
		self.k = k

	def train(self, anchor_points, anchor_labels):
		"""
		:param anchor_points: a 2D numpy array, in which each row is
							  a datapoint, without its label, to be used
							  for one of the anchor points

		:param anchor_labels: a list in which the i'th element is the correct label
							  of the i'th datapoint in anchor_points

		Does not return anything; simply stores anchor_labels and the
		_features_ of anchor_points.
		"""
		self.num_samples = len(anchor_points)
		self.features = [None] * self.num_samples
		for i in range(self.num_samples):
			self.features[i] = np.append(anchor_points[i], anchor_labels[i])


	def predict(self, x):
		"""
		Given a single data point, x, represented as a 1D numpy array,
		predicts the class of x by taking a plurality vote among its k
		nearest neighbors in feature space. Resolves ties arbitrarily.

		The K nearest neighbors are determined based on Euclidean distance
		in _feature_ space (so be sure to compute the features of x).

		Returns the label of the class to which x is predicted to belong.
		"""
		result_map = [None] * self.num_samples
		label = 0
		for i in range(self.num_samples):
			total_sum = 0
			for j in range(len(x)):
				total_sum += ((self.features[i][j] - x[j]) ** 2)
			euc_distance = np.sqrt(total_sum)
			result_map[i] = (euc_distance, self.features[i][len(x)])
		neighbours = [None] * self.k
		for neighbour in range(self.k):
			closest_neighbour = min(result_map)
			result_map.remove(closest_neighbour)
			neighbours[neighbour] = closest_neighbour
		label = max(set(neighbours), key=neighbours.count)
		if label is not None:
			return label[1]
		else:
			return 0
		

	def evaluate(self, datapoints, labels):
		"""
		:param datapoints: a 2D numpy array, in which each row is a datapoint.
		:param labels: a 1D numpy array, in which the i'th element is the
					   correct label of the i'th datapoint.

		Returns the fraction (between 0 and 1) of the given datapoints to which
		predict(.) assigns the correct label
		"""
		num_correct = 0
		for i in range(len(datapoints)):
			if self.predict(datapoints[i]) == labels[i]:
				num_correct += 1
		return num_correct / len(datapoints)
		