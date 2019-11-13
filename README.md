# Fashion-MNIST
Fashion MNIST Classification using Bayes and KNN Classifier + Feature dimension reduction using PCA and LDA


\section*{ML and Byes Rule Classification}

We assume that the density of the data points is a multi variable Gaussian density. The probability density function is:\\




$$p(\mathbf{x} \mid \mathbf{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \lvert\Sigma\rvert}} \exp{ \left( -\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \Sigma^{-1} (\mathbf{x} - \mathbf{\mu}) \right)}$$

Which d is the feature size. For the fashion MNIST the feature size is equal to the image dimensions which id:

$$d = 28 \times 28 = 784$$
