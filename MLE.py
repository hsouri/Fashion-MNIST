import numpy as np


class MLE:

    def __init__(self, data):
        self.data = data

    def mean(self):
        my_mean = np.sum(self.data, axis = 0)/self.data.shape[0]
        numpy_mean = np.mean(self.data, axis=0)

        if np.array_equal(my_mean, numpy_mean):
            return my_mean.reshape(my_mean.shape[-1], 1)
        else:
            print("Mean is not correct!")
            return None

    def covariance(self):
        my_cov = np.matmul(self.data.T - self.mean(), (self.data.T - self.mean()).T)/self.data.shape[0]
        numpy_cov = np.cov (self.data.T) * (self.data.shape[0]-1)/self.data.shape[0]

        if sum(sum(abs(my_cov - numpy_cov))) < 10e5:
            return my_cov
        else:
            print("Covariance is not correct!")
            return None
