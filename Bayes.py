import numpy as np
from MLE import MLE
from sklearn import metrics
import math
from matplotlib import pyplot as plt
import itertools


class Bayes:

    def __init__(self, train_data, train_labels, test_data, test_labels, name, file):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_of_classes = max(train_labels) - min(train_labels) + 1
        self.priors = []; self.means = []; self.covariances = []; self.new_covariances = []
        self.new_covariances_inverse = []; self.new_covariances_det = []
        self.name = name
        self.file = file

    def train(self):
        for c in range(self.num_of_classes):
            self.priors.append(self.class_data(c).shape[0]/self.test_data.shape[0])
            self.means.append(MLE(self.class_data(c)).mean())
            self.covariances.append(MLE(self.class_data(c)).covariance())
        return self

    def test(self):
        scores = []
        self.covariance_regularization()
        for i in range(self.test_data.shape[0]):
            scores.append(self.discriminant(i))
            if i % 100 == 0:
                print (' Test data ' + str(i) + ' out of ' + str(self.test_data.shape[0]))
        predicted_labels = np.argmax(scores, axis=1)
        np.save(self.name + '_bayes_predicted_labels.npy', predicted_labels)
        accuracy = 100 * metrics.accuracy_score(self.test_labels, predicted_labels)
        self.write_results(accuracy, self.file)
        print("Accuracy:", accuracy)
        self.confusion_matrix(predicted_labels)

    def class_data(self, c):
        mask = np.zeros((self.num_of_classes,), dtype=bool)
        mask[c] = True
        return self.train_data[mask[self.train_labels], :]

    def covariance_regularization(self, _lambda=0.01, _gamma=0.1):
        m_p = np.zeros((self.train_data.shape[1], self.train_data.shape[1]))
        for c1 in range(self.num_of_classes):
            m_p += self.class_data(c1).shape[0] * self.covariances[c1]
        m_p = m_p / self.train_data.shape[0]

        for c in range(self.num_of_classes):
            lambda_ = _lambda
            m_pool = (1 - lambda_) * self.covariances[c] + lambda_ * m_p
            s = np.matrix.trace(m_pool) / self.train_data.shape[1]
            m_rda = (1 - _gamma) * m_pool + _gamma * s * np.identity(self.train_data.shape[1])

            self.new_covariances.append(m_rda)
            self.new_covariances_inverse.append(np.linalg.inv(self.new_covariances[c]))
            self.new_covariances_det.append(np.linalg.det(self.new_covariances[c]))

    def discriminant(self, i):
        results = []
        for c in range(self.num_of_classes):
            prior = self.class_data(c).shape[0]/self.train_data.shape[0]
            x = self.test_data[i] - self.means[c].flatten()
            arg1 = np.matmul(x.T, self.new_covariances_inverse[c])
            arg2 = np.matmul(arg1, x)
            y = 2 * math.log(prior) - arg2 - math.log(self.new_covariances_det[c])
            results.append(y)
        return results

    def confusion_matrix(self, predicted_labels):
        confusion = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.test_data.shape[0]):
            confusion[self.test_labels[i], predicted_labels[i]] += 1
        np.save(self.name + 'bayes_confusion.npy', confusion)
        self.plot_confusion_matrix(confusion)

    def plot_confusion_matrix(self, confusion_matrix, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

        fig = plt.figure()
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.num_of_classes)
        plt.xticks(tick_marks, np.arange(10))
        plt.yticks(tick_marks, np.arange(10))

        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.name + '_bayes.jpg')

    def write_results(self, accuracy, file):
        f = open(file, 'a')
        f.write('\n' + self.name + '_bayes: ' + 'accuracy is ' + str(accuracy) + '\r\n')
        f.close()