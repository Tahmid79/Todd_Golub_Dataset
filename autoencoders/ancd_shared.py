import keras
import sklearn
from sklearn import svm
from keras.layers import Dense ,  Input , LeakyReLU
from keras.models import  Model
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy
from sklearn.utils import  shuffle
from sklearn.naive_bayes import GaussianNB

from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap

from data.preprocess import  ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num

selection =  SelectKBest(k=50)


(features1 , labels1) = ft_lbls_num()
features1 = selection.fit_transform(features1 , labels1)


(features2, labels2) = preprocess_ft_lbls_num()
features2 = selection.fit_transform(features2 , labels2)


input = Input(shape=(50, ))
hd =  LeakyReLU()( Dense(20)(input)  )
output =  Dense(50 ,  activation='softmax')(hd)

model = Model( input , output)
encoder= Model( input, hd )

model.compile(loss='mean_squared_error', optimizer='sgd')



for i in range(500):

    if ( i%2==0):
        model.fit(features1 ,  features1 ,  batch_size=len(features1) , epochs=10 , validation_split=0.9 )

    if (i%2==1):
        model.fit(features2, features2, batch_size=len(features2), epochs=10 , validation_split=0.9  )



rd_dim = []

rd_dim2 = []

for item in features1:
    lst = []
    lst.append(item)
    lst =  numpy.asarray(lst).astype(numpy.float32)
    pred = encoder.predict(lst)
    prediction = pred[0].tolist()
    rd_dim.append(prediction)

for item in features2:
    lst = []
    lst.append(item)
    lst =  numpy.asarray(lst).astype(numpy.float32)
    pred = encoder.predict(lst)
    prediction = pred[0].tolist()
    rd_dim2.append(prediction)


rd_dim = numpy.asarray(rd_dim , dtype=numpy.float32)
rd_dim2 = numpy.asarray(rd_dim2 , dtype=numpy.float32)



features = numpy.concatenate((rd_dim , rd_dim2))
labels = numpy.concatenate((labels1 , labels2))



scores = []

clf = GaussianNB()

K=5
cv = KFold(n_splits=K , shuffle=True)

for i in range(100):

    rd_dim , labels1 =  shuffle( rd_dim  , labels1)

    for train, test in cv.split(rd_dim):

        features_train = numpy.concatenate(( rd_dim[train] , rd_dim2  ))
        features_test = rd_dim[test]

        labels_train = numpy.concatenate(( labels1[train] , labels2))
        labels_test = labels1[test]

        clf.fit(features_train , labels_train)

        pred = clf.predict(features_test)
        score = accuracy_score(labels_test , pred)
        scores.append(score)


average_score = sum(scores)/(len(scores))
print(scores)
print("Average Score = " ,  round( average_score  ,  5)  )
print('Standard Deviation = ' ,  numpy.std(scores))


















