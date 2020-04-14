import numpy
import sklearn
from sklearn import svm

from ptc.dt1_prtc import preprocess_ft_lbls

clf = svm.SVC()

(features , labels) =  preprocess_ft_lbls()

features = numpy.asarray(features).astype(numpy.float32)
labels = numpy

print(features.shape)
print(len(labels))


