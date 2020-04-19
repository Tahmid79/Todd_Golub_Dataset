import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import  pyplot
#from mlxtend.plotting import plot_decision_regions
from sklearn.manifold import MDS ,  LocallyLinearEmbedding

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA

from data.preprocess import features_preprocess ,features_test_preprocess , labels_preprocess

labels_all = labels_preprocess()

labels_train = labels_all[:38]
labels_test = labels_all[38:]

features_train = features_preprocess()
features_test = features_test_preprocess()



print("Dimensionality = ",len(features_train[1]))

"""
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)
"""

embedding  = LocallyLinearEmbedding(n_components=10 , n_neighbors=5)


features_train = embedding.fit_transform(features_train , labels_train)
features_test = embedding.transform(features_test)


clf = svm.SVC(kernel='linear')

clf.fit(features_train , labels_train)


pred = clf.predict(features_test)
score = accuracy_score(labels_test , pred)
print(score)
















