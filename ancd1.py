import keras
from keras.layers import Dense , Input
from keras.models import Model
from keras.losses import  mean_squared_error

import numpy
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from preprocess import features_test_preprocess , features_preprocess , labels_preprocess_num , labels_preprocess


features = numpy.concatenate((features_preprocess() ,  features_test_preprocess()))
labels = labels_preprocess_num()



features_trn = features[:60]
labels_trn = labels[:60]


features_tst = features[60:]
labels_tst = labels[60:]


selection  = SelectKBest(k=50)
features_trn = selection.fit_transform(features_trn , labels_trn)
features_tst = selection.transform(features_tst)
features = selection.transform(features)



input_layer = Input(shape=(50, ))
hd1 = Dense(20 )(input_layer)
lky=  keras.layers.LeakyReLU()(hd1)
output_layer = Dense(50 , activation='softmax')(hd1)

model = Model(input_layer , output_layer)
encoder = Model(input_layer ,  hd1)

model.compile(optimizer='sgd' , loss='mean_squared_error')
model.fit( features_trn ,  features_trn , epochs=1000 , batch_size=60 )

rd_dim = []

for item in features:
    lst = []
    lst.append(item)
    lt = numpy.asarray(lst , dtype=numpy.float32)
    pred = encoder.predict(lt)
    prediction = pred[0].tolist()
    rd_dim.append(prediction)


print(rd_dim)

rd_dim = numpy.asarray(rd_dim , dtype=numpy.float32)

clf  = svm.SVC(kernel='rbf')
clf.fit(rd_dim[:60] ,  labels[:60])

pred = clf.predict(rd_dim[60:])

accuracy = accuracy_score(pred , labels[60:])
print(accuracy)



















