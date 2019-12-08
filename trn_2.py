import pandas
import numpy
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.decomposition import PCA
import math

from preprocess import features_preprocess , features_test_preprocess , labels_preprocess

labels = labels_preprocess()
features_1 = features_preprocess()
features_2 = features_test_preprocess()

features = numpy.concatenate((features_1 ,features_2) )

clf1 = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='rbf')
clf3 = GaussianNB()
clf4 = KNeighborsClassifier(n_neighbors=5)
classifiers = [clf1 , clf2  ,clf3 , clf4]
classifier_name = ['SVM Linear' , "SVM RBF" , "Gaussian NB" , "KNN-5"]
average_scores = []

for i in range(4):

    clf= classifiers[i]

    K = 10

    pca = PCA(n_components=10)

    cv = KFold(n_splits=K)

    scores = []

    for train , test in cv.split(features):
        train_feat = features[train]
        test_feat = features[test]


        train_labels = [ labels[i] for i in train  ]
        test_labels = [ labels[i] for i in test]

        pca.fit(train_feat)

        train_feat = pca.transform(train_feat)
        test_feat= pca.transform(test_feat)

        clf.fit(train_feat , train_labels)
        prediction = clf.predict(test_feat)
        scr = accuracy_score(prediction , test_labels)

        scores.append(scr)


    #for i in range(K):
        #print('Iteration ',i+1 ,' =' , scores[i] )

    average_score = sum(scores)/K
    scores_rounded = [round(scores[i] , 3) for i in range(K) ]

    print("K = ",K)
    print("Classifier " + classifier_name[i] )
    print(scores_rounded)
    print('Average Score = ' , round(average_score, 4)  )
    print()
    average_scores.append(average_score)

    #print(len(features))

average_scores = [average_scores[i]*100 for i in range(len(average_scores))]
left = [1 ,2 ,3,4]
plt.bar(left , average_scores , tick_label = classifier_name , width=0.8 )
plt.xlabel = 'Classifier'
plt.ylabel = 'Average Accuracy K=10'
plt.show()

