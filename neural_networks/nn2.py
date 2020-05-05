import numpy
import keras
import sklearn
from sklearn import svm
from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from data.preprocess import ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num


(features ,  labels) = ft_lbls_num()

selection = SelectKBest(k=15)
features = selection.fit_transform(features , labels)

input = Input(shape=(15 , ))
hd1 =   Dense(8, activation='relu')(input)
hd2 = Dense(5 , activation='tanh')(hd1)
output=  Dense(1 , activation='softmax')(hd2)

model = Model(input , output)

model.compile(loss='mean_squared_error' , optimizer='sgd')
model.fit(features , labels , batch_size=len(features) , epochs=100 ,  validation_split=0.9  )


cv = KFold(n_splits=5)



scores = []


for i in range(100):

    features, labels = shuffle(features, labels)

    for train, test in cv.split(features):
        features_trn = features[train]
        features_test = features[test]

        labels_trn = labels[train]
        labels_test = labels[test]

        #model.fit(features_trn , labels_trn , batch_size=len(features_trn) , epochs=100 )

        pred = model.predict(features_test)
        score = accuracy_score(pred , labels_test)
        scores.append(score)

print(scores)
average_score = sum(scores)/(len(scores))
print('Average Score = ' , round( average_score ,  5  )  )
print('Standard Deviation = ' , numpy.std(  scores ) )