import numpy
import sklearn
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.decomposition import  PCA
from sklearn.metrics import  accuracy_score
from sklearn.utils import  shuffle

import sys
sys.path.append('../')



#from ptc.dt1_prtc import preprocess_ft_lbls , preprocess_ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num ,  preprocess_ft_lbls

K = 5
cv = KFold(n_splits=K, shuffle=True)

(features , labels) = preprocess_ft_lbls_num()
#labels = numpy.asarray(labels, dtype=numpy.float32)

variance_ratio = []
scores = []

#X, y = shuffle(X, y)

for train, test in cv.split(features):

    features_train = features[train]
    features_test = features[test]

    labels_train = labels[train]
    labels_test = labels[test]

    pca = PCA(n_components=10)
    pca.fit(features_train)

    features_train = pca.transform(features_train)
    features_test = pca.transform(features_test)


    clf = svm.SVC(kernel='linear')

    clf.fit(features_train , labels_train)


    pred = clf.predict(features_test)
    score = accuracy_score(labels_test , pred)
    scores.append(score)

    print( pca.explained_variance_ratio_ )
    sm = pca.explained_variance_ratio_.sum()
    variance_ratio.append(sm)


print(variance_ratio)
print(scores)


"""

arr1 = [1, 2, 3, 4, 5]
arr2 = [2, 3, 3, 4, 4]
labl = [0, 1, 1, 0, 0]
color= ['red' if l == 0 else 'green' for l in labl]
plt.scatter(arr1, arr2, color=color)

"""

"""

from sklearn.model_selection import LeaveOneOut 
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)


"""















