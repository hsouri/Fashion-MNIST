# Fashion-MNIST
Fashion MNIST Classification using Bayes and KNN Classifier + Feature dimension reduction using PCA and LDA

This is a Python implementation of Bays and K-nn classifer plus PCA and LDA dimension reduction on Fashion MNIST fataset.  It’s consisted of mnistreader.py, Bayes.py class, KNN.py class, and Driver.py.  

Note: Please download and copy fashion train set into the fashion folder of the data folder. Link: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

The results are saved into out.txt file.

# Usage:


The training can be done by running the main function in the Driver.py.

How to read fashion mnist: 

traindata, trainlabels = mnistreader.loadmnist(’data/fashion’, kind=’train’)
testdata, testlabels = mnistreader.loadmnist(’data/fashion’, kind=’t10k’)

# Bayes Train and Test:

bayesclassifier = Bayes(traindata, trainlabels, testdata, testlabels, ’Name’, output file)bayesclassifier.train().test()


# K-NN Train and Test:

knnclassifier = KNN(traindata, trainlabels, testdata, testlabels, k , ’Name’, out put file)
knnclassifier.train().test()

# Train all models:
python Driver.py
