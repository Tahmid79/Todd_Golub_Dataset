import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
#from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle

import sys
sys.path.append('../')



from data.preprocess import features_preprocess ,features_test_preprocess , labels_preprocess , labels_preprocess_num
from data.preprocess_2nd import  preprocess_ft_lbls_num
from data.preprocess import  ft_lbls_num


scores = []

pca = PCA(n_components=10)


(features1 ,  labels1) = ft_lbls_num()
(features2 ,  labels2) = preprocess_ft_lbls_num()


features1 = pca.fit_transform(features1, labels1)

features2 = pca.fit_transform(features2, labels2)


K=5
cv = KFold(n_splits=K, shuffle=True)

features = numpy.concatenate( (features1, features2) )
labels = numpy.concatenate( (labels1, labels2  ) )

for i in range(100):

    features , labels = shuffle(features , labels)

    for train, test in cv.split(features):

        features_train = features[train]
        features_test = features[test]

        labels_train = labels[train]
        labels_test = labels[test]


        clf = svm.SVC(kernel='rbf' )

        clf.fit(features_train , labels_train)


        pred = clf.predict(features_test)
        score = accuracy_score(labels_test , pred)
        scores.append(score)


print()
print(scores)
print('Average Score = ' , round( sum(scores)/(5 * 100) ,  5  )  )























