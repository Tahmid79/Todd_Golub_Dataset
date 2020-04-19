import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils import  shuffle

from data.preprocess import ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num

(features , labels) = ft_lbls_num()

clf = svm.SVC(kernel='linear')

K = 5
cv  =KFold(n_splits=K, shuffle=True)

pca = PCA(n_components=20)


scores = []

variance_ratio1 = []
variance_ratio2 =[]

selection = SelectKBest(k=100)

for train, test in cv.split(features):
    features_trn = features[train]
    features_test = features[test]

    labels_trn = labels[train]
    labels_test = labels[test]

    features_trn = selection.fit_transform(features_trn, labels_trn)
    features_test =selection.transform(features_test)

    features_trn = pca.fit_transform(features_trn, labels_trn)
    features_test = pca.transform(features_test)

    clf.fit(features_trn, labels_trn)
    result = clf.predict(features_test)
    score = accuracy_score(result, labels_test)
    scores.append(score)
    vrnc = pca.explained_variance_ratio_
    variance_ratio1.extend(vrnc)


average_score = sum(scores)/K



(features, labels) = preprocess_ft_lbls_num()

scores = []

for train, test in cv.split(features):
    features_trn = features[train]
    features_test = features[test]

    labels_trn = labels[train]
    labels_test = labels[test]

    features_trn = selection.fit_transform(features_trn, labels_trn)
    features_test = selection.transform(features_test)

    features_trn = pca.fit_transform(features_trn, labels_trn)
    features_test = pca.transform(features_test)

    clf.fit(features_trn, labels_trn)
    result = clf.predict(features_test)
    score = accuracy_score(result, labels_test)
    scores.append(score)
    vrnc = pca.explained_variance_ratio_
    variance_ratio2.extend(vrnc)


print('Variance Ratio 1')
print(variance_ratio1)
print(len(variance_ratio1))

print()
print('Variance Ratio 2')
print(variance_ratio2)
print(len(variance_ratio2))

print()

print([  round(item, 7) for item in variance_ratio1[30:45] ])
print([  round(item, 7) for item in variance_ratio2[30:45] ])

from pca.reg import best_fit

best_fit(variance_ratio1 , variance_ratio2)










