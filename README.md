# Fashion-MNIST
Fashion MNIST Classification using Bayes and KNN Classifier + Dimension reduction using PCA and LDA

This is a Python implementation of Bayes and K-nn classifer plus PCA and LDA dimension reduction on Fashion MNIST dataset.  It’s consisted of mnistreader.py, Bayes.py class, KNN.py class, and Driver.py.  

Note: Please download and copy fashion train set into the fashion folder of the data folder. Link: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

The results are saved into 'out.txt' file.

# Getting Started
Clone this repository with the following command:

```shell
$ git clone https://github.com/hsouri/Fashion-MNIST
```

# Usage

The training can be done by running the main function in the Driver.py.

# Train all models:

```shell
$ python Driver.py

```

How to read fashion mnist: 

```shell
traindata, trainlabels = mnistreader.loadmnist(’data/fashion’, kind=’train’)
testdata, testlabels = mnistreader.loadmnist(’data/fashion’, kind=’t10k’)
```

# Bayes Train and Test:

```shell
bayesclassifier = Bayes(traindata, trainlabels, testdata, testlabels, ’Name’, output file)
bayesclassifier.train().test()
```


# K-NN Train and Test:

```shell
knnclassifier = KNN(traindata, trainlabels, testdata, testlabels, k , ’Name’, out put file)
knnclassifier.train().test()
```


# PCA and LDA:

Take a look at main function in the Driver.py to see how to apply PCA and LDA to the feature vector.


