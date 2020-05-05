import matplotlib
import numpy
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA



from data.preprocess import features_preprocess , features_test_preprocess , labels_preprocess , labels_preprocess_num
from data.preprocess_2nd import  preprocess_ft_lbls_num

labels = labels_preprocess()
features = numpy.concatenate( (  features_preprocess() ,features_test_preprocess()  ) )

#(features , labels) = preprocess_ft_lbls_num()

K= 5
cv = KFold(n_splits=K, shuffle=True)

pca5 = PCA(n_components=5)
pca10  = PCA(n_components=10)
pca15 = PCA(n_components=15)
pca20 =PCA(n_components=20)
pca25 = PCA(n_components=25)

clf = svm.LinearSVC()

average_scores = []

component_num = [5,10,15,20,25]

pca_list = [pca5 , pca10 , pca15 , pca20 , pca25]

for i in range(len(pca_list)):

    pca = pca_list[i]
    scores = []

    for train , test in cv.split(features):
        features_train = features[train]
        features_test = features[test]

        labels_train = [labels[i] for i in train]
        labels_test = [labels[i] for i in test]


        features_train = pca.fit_transform(features_train)
        features_test = pca.transform(features_test)

        clf.fit(features_train , labels_train)
        result = clf.predict(features_test)
        score = accuracy_score(labels_test , result)
        scores.append(score)

    scores = [ round(score*100,2)  for score in scores]
    average = sum(scores)/K
    average_scores.append(average)

    print('Principal Components = ', component_num[i] )
    print(scores)
    print('\n')

matplotlib.use('TkAgg')
fig, ax = plt.subplots()

ax.set_xlabel('No of Principal Components')
ax.set_ylabel( 'Average Accuracy K=5')

component_num_str = ['5','10','15','20','25']


plt.bar(list(range(1,6)) , average_scores , tick_label = component_num_str , width=0.5 )


plt.show()






