import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import  shuffle

from data.preprocess import ft_lbls_num , ft_lbls
from data.preprocess_2nd import preprocess_ft_lbls_num

(features , labels) = ft_lbls()



K = 5
cv  = KFold(n_splits=K, shuffle=True)


clf = GaussianNB()

scores = []

for i in range(100):

    features , labels = shuffle(features, labels)

    for train, test in cv.split(features):
        features_trn = features[train]
        features_test = features[test]

        labels_trn = [labels[item] for item in train]
        labels_test = [labels[item] for item in test]


        clf.fit(features_trn, labels_trn)
        result = clf.predict(features_test)
        score = accuracy_score(result, labels_test)
        scores.append(score)


print()
print(scores)
print('Average Score = ' , round( sum(scores)/(len(scores)) ,  5  )  )
print('Standard Deviation = ' , numpy.std(  scores ) )


ft1 , lb1 = preprocess_ft_lbls_num()

print(ft1.shape)
















