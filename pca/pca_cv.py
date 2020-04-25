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

import sys
sys.path.append('../')



from data.preprocess import features_preprocess ,features_test_preprocess , labels_preprocess , labels_preprocess_num
from data.preprocess_2nd import  preprocess_ft_lbls_num
from data.preprocess import  ft_lbls_num


scores =[]
variance_ratio = []


#features = numpy.concatenate( ( features_preprocess() , features_test_preprocess() ) )
#labels = labels_preprocess_num()

(features ,  labels) = ft_lbls_num()


variance_ratio = []
scores = []

K=10
cv = KFold(n_splits=K, shuffle=True)

selection = SelectKBest(k=50)

pca = PCA(n_components=10)

clf = svm.SVC(kernel='linear')

for train, test in cv.split(features):

    features_train = features[train]
    features_test = features[test]

    labels_train = labels[train]
    labels_test = labels[test]

    pca.fit(features_train)

    features_train = pca.transform(features_train)
    features_test = pca.transform(features_test)


    clf.fit(features_train , labels_train)


    pred = clf.predict(features_test)
    score = accuracy_score(labels_test , pred)
    scores.append(score)

    sm = pca.explained_variance_ratio_.sum()
    variance_ratio.append(sm)


print(variance_ratio)
print(scores)
print("Average Score = " ,sum(scores)/K  )














