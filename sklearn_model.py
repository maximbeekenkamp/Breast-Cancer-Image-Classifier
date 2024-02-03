import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from supervisedlearner import SupervisedLearner

class SKLearnModel(SupervisedLearner):
    def __init__(self, feature_funcs):
        super().__init__(feature_funcs)
        # initialize field for linear regression model
        self.linear_regressor = None
        self._trained = False

    def train(self, X, Y):
        """

        :param X: a 2D numpy array where each row represents a datapoint
        :param Y: a 1D numpy array where i'th element is the label of the corresponding datapoint in X
        :return: None

        """

        self._trained = True
        self.linear_regressor = LinearRegression()
        self.linear_regressor.fit(X, Y)

    def predict(self, x):
        """
        :param x: a 1D numpy array representing a single datapoints
        :return: prediction of the linear regression model

        """

        assert self._trained

        return self.linear_regressor.predict(x.reshape(1,-1))

    def evaluate(self, datapoints, labels):
        """

        :param datapoints: a 2D numpy array where each row represents a datapoint
        :param labels: a 1D numpy array where i'th element is the label of the corresponding datapoint in datapoints
        :return: a tuple with the Mean Squared Error of the predictions over datapoints relative to labels
                 and the R Square (R2) Score

        """

        assert self._trained

        y_pred = self.linear_regressor.predict(datapoints)
        return (metrics.mean_squared_error(labels, y_pred), metrics.r2_score(labels, y_pred))
