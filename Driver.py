import mnist_reader
from Bayes import Bayes
from KNN import KNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def main():
    file = 'out.txt'
    train_data, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
    train_data = 5 * (train_data / 255 - 0.5) / 0.5
    test_data, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')
    test_data = 5 * (test_data / 255 - 0.5) / 0.5
    num_of_classes = max(test_labels) - min(test_labels) + 1

    # raw pixel bayes train and test
    bayes_classifier = Bayes(train_data, train_labels, test_data, test_labels, 'raw', file)
    bayes_classifier.train().test()
    # raw pixel 1NN train and test
    knn_classifier = KNN(train_data, train_labels, test_data, test_labels, 1, 'raw', file)
    knn_classifier.train().test()

    # k-NN with different values of k
    for k in range(10):
        knn_classifier = KNN(train_data, train_labels, test_data, test_labels, k + 1, 'RAW_', file)
        knn_classifier.train().test()

    # PCA with different components from 100 to 700 with step 100
    for i in range(1, 8):
        pca = PCA(n_components=100*i)
        pca.fit(train_data)
        pca_train_data = pca.transform(train_data)
        pca_test_data = pca.transform(test_data)
        bayes_classifier = Bayes(pca_train_data, train_labels, pca_test_data, test_labels, 'PCA_'
                                 + str(pca.n_components_) + '_components', file)
        bayes_classifier.train().test()

        for j in range(5):
            knn_classifier = KNN(pca_train_data, train_labels, pca_test_data, test_labels, 10*j + 1, 'PCA_'
                                    + str(pca.n_components_) + '_components', file)
            knn_classifier.train().test()

    # PCA and LDA which different values of components between 1 and 9 with step 1
    for components in range(1, 10):
        pca = PCA(n_components=components)
        pca.fit(train_data)
        pca_train_data = pca.transform(train_data)
        pca_test_data = pca.transform(test_data)
        bayes_classifier = Bayes(pca_train_data, train_labels, pca_test_data, test_labels, 'PCA_'
                                 + str(pca.n_components_) + '_components', file)
        bayes_classifier.train().test()

        for j in range(10):
            knn_classifier = KNN(pca_train_data, train_labels, pca_test_data, test_labels, 10*j + 1, 'PCA_'
                                 + str(pca.n_components_) + '_components', file)
            knn_classifier.train().test()

        lda = LDA(n_components=components)
        lda.fit(train_data, train_labels)
        lda_train_data = lda.transform(train_data)
        lda_test_data = lda.transform(test_data)
        bayes_classifier = Bayes(lda_train_data, train_labels, lda_test_data, test_labels, 'LDA_'
                                 + str(lda.n_components) + '_components', file)
        bayes_classifier.train().test()

        for j in range(10):
            knn_classifier = KNN(lda_train_data, train_labels, lda_test_data, test_labels, 10*j + 1, 'LDA_'
                                 + str(lda.n_components) + '_components', file)
            knn_classifier.train().test()


if __name__ == '__main__':
    main()