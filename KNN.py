import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import itertools


class KNN:

    def __init__(self, train_data, train_labels, test_data, test_labels, neighbors, name, file):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_of_classes = max(train_labels) - min(train_labels) + 1
        self.neighbors = neighbors
        self.knn = KNeighborsClassifier(n_neighbors=neighbors)
        self.name = name
        self.file = file

    def train(self):
        self.knn.fit(self.train_data, self.train_labels)
        return self

    def test(self):
        predicted_labels = self.knn.predict(self.test_data)
        np.save(self.name + '_' + str(self.neighbors) + 'nn_predicted_labels.npy', predicted_labels)
        accuracy = 100 * metrics.accuracy_score(self.test_labels, predicted_labels)
        self.write_results(accuracy, self.file)
        print("Accuracy:", accuracy),
        self.confusion_matrix(predicted_labels)

    def confusion_matrix(self, predicted_labels):
        confusion = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.test_data.shape[0]):
            confusion[self.test_labels[i], predicted_labels[i]] += 1
        np.save(self.name + '_' + str(self.neighbors) + 'nn_confusion.npy', confusion)
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
        plt.savefig(self.name + '_' + str(self.neighbors) + 'nn.jpg')

    def write_results(self, accuracy, file):
        f = open(file, 'a')
        f.write('\n' + self.name + '_' + str(self.neighbors) + 'nn: ' + 'accuracy is ' + str(accuracy) + '\r\n')
        f.close()