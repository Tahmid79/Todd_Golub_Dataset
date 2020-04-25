import sklearn
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import  PCA
from matplotlib import pyplot



x = numpy.array([1, 3, 5, 7])
y = numpy.array([ 6, 3, 9, 5 ])

m , b = numpy.polyfit(x , y, 1)

pyplot.plot(x, y , 'o')

pyplot.plot(x, m * x + b)
pyplot.show()


def best_fit(x , y ):
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    m, b = numpy.polyfit(x, y, 1)

    pyplot.plot(x, y, 'o')

    pyplot.plot(x, m * x + b)
    pyplot.show()


