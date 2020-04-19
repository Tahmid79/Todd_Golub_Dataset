import numpy
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

import sys
sys.path.append('../')

#from preprocess import features_preprocess ,  features_test_preprocess ,  labels_preprocess_num
from data.preprocess import features_preprocess ,  features_test_preprocess ,  labels_preprocess_num

"""
features = [ [1,2], [1,3],[1,6], [ 1,1] , [5,6] , [6,7], [3,7] , [1,2] , [7,8] , [4,1] ]
labels = [ [1],[0],[1],[0] ,[1],[0],[1],[0] , [1] , [0] ]

#features = numpy.concatenate((features_preprocess() , features_test_preprocess() ))
#labels = labels_preprocess_num()

features = numpy.asarray(features, dtype=numpy.float32)
labels = numpy.asarray(labels , dtype=numpy.float32)
"""

features = numpy.concatenate((features_preprocess() , features_test_preprocess()))
labels = labels_preprocess_num()

selection = SelectKBest(k=1)


clf = svm.SVC(kernel='linear')

K = 3

cv = KFold(n_splits=K)

scores = []

for train ,  test in cv.split(features):

    ft_trn = features[train]
    ft_tst = features[test]

    lb_ft = labels[train]
    lb_tst = labels[test]

    ft_trn = selection.fit_transform(ft_trn , lb_ft)
    ft_tst = selection.transform(ft_tst)

    clf.fit(ft_trn , lb_ft)

    prd = clf.predict(ft_tst)

    score = accuracy_score(prd , lb_tst)
    score  = float(score)
    scores.append(score)

type(scores)
print(scores)











