#Data Rescaling
Your preprocessed data may contain attributes with a mixtures of scales for various quantities such as dollars, kilograms and sales volume.
Many machine learning methods expect or are more effective if the data attributes have the same scale. Two popular data scaling methods are normalization and standardization.

#Data Normalization
Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
It is useful to scale the input attributes for a model that relies on the magnitude of values, such as distance measures used in k-nearest neighbors and in the preparation of coefficients in regression.
The example below demonstrate data normalization of the Iris flowers dataset.

# Normalize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
y = iris.target
# normalize the data attributes
normalized_X = preprocessing.normalize(X)


# Normalize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
y = iris.target
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
For more information see the normalize function in the API documentation.

#Data Standardization
Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance).
It is useful to standardize attributes for a model that relies on the distribution of attributes such as Gaussian processes.
The example below demonstrate data standardization of the Iris flowers dataset.

# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the Iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data and target attributes
X = iris.data
y = iris.target
# standardize the data attributes
standardized_X = preprocessing.scale(X)

# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the Iris dataset
iris = load_iris()
print(iris.data.shape)

# separate the data and target attributes
X = iris.data
y = iris.target
# standardize the data attributes
standardized_X = preprocessing.scale(X)
For more information see the scale function in the API documentation.

Fonte: https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/