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

(features , labels) = preprocess_ft_lbls_num()

clf = svm.SVC(kernel='linear')

K = 3
cv  =KFold(n_splits=K, shuffle=True)

pca = PCA(n_components=10)


average_scores = []


for i in range(100):
    scores = []
    features , labels = shuffle(features, labels)

    for train, test in cv.split(features):
        features_trn = features[train]
        features_test = features[test]

        labels_trn = labels[train]
        labels_test = labels[test]

        features_trn = pca.fit_transform(features_trn, labels_trn)
        features_test = pca.transform(features_test)

        clf.fit(features_trn, labels_trn)
        result = clf.predict(features_test)
        score = accuracy_score(result, labels_test)
        scores.append(score)


    average_score = sum(scores)/K
    average_scores.append(average_score)

print(average_scores)
print(numpy.average(average_scores))











