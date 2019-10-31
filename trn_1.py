import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import  pyplot
from mlxtend.plotting import plot_decision_regions

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA

from preprocess import features_preprocess ,features_test_preprocess , labels_preprocess

labels_all = labels_preprocess()

labels_train = labels_all[:38]
labels_test = labels_all[38:]

features_train = features_preprocess()
features_test = features_test_preprocess()

"""
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)
"""

pca = PCA(n_components=10)
pca.fit(features_train)

features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


clf = svm.SVC(kernel='linear')

clf.fit(features_train , labels_train)


pred = clf.predict(features_test)
score = accuracy_score(labels_test , pred)

print("Score = " , score )
print()
print(pca.explained_variance_ratio_ )
print()
print(pca.explained_variance_ratio_.sum() )














