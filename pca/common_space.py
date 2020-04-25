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

print(features1.shape)
print(features2.shape)

print()
features1 = pca.fit_transform(features1, labels1)

features2 = pca.fit_transform(features2, labels2)

print(features1.shape)
print(features2.shape)

K=5
cv = KFold(n_splits=K, shuffle=True)

features = numpy.concatenate( (features1, features2) )
labels = numpy.concatenate( (labels1, labels2  ) )

clf = svm.SVC(kernel='rbf')

for i in range(100):

    features1 , labels1 = shuffle(features1 , labels1)

    for train, test in cv.split(features1):

        features_train = numpy.concatenate(( features1[train] , features2  ))
        features_test = features1[test]

        labels_train = numpy.concatenate(( labels1[train] ,  labels2 ))
        labels_test = labels1[test]

        clf.fit(features_train , labels_train)

        pred = clf.predict(features_test)
        score = accuracy_score(labels_test , pred)
        scores.append(score)


print()
print(scores)
print('Average Score = ' , round( sum(scores)/(len(scores)) ,  5  )  )
print('Standard Deviation = ' , numpy.std(  scores ) )























