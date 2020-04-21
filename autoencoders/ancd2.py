import keras
import sklearn
from sklearn import svm
from keras.layers import Dense ,  Input , LeakyReLU
from keras.models import  Model
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy


import sys
sys.path.append('./data/')


#from preprocess import features_test_preprocess , features_preprocess , labels_preprocess_num
from data.preprocess_2nd import  preprocess_ft_lbls_num

#features = numpy.concatenate((features_preprocess() ,  features_test_preprocess() ))
#labels = labels_preprocess_num()

(features , labels) = preprocess_ft_lbls_num()
labels = numpy.asarray(labels , dtype=numpy.float32)

input = Input(shape=(50,))
hd1 = Dense(20)(input)
leaky = LeakyReLU()(hd1)
output = Dense(50, activation='softmax')(hd1)

model = Model(input, output)
encoder = Model(input, hd1)

model.compile(optimizer='sgd', loss='mean_squared_error')



K=5
cv = KFold(n_splits=K, shuffle=True)
scores =  []

for train , test in cv.split(features):

    selection = SelectKBest(k=50)
    features_trn = selection.fit_transform(features[train] , labels[train])
    features_tst = selection.transform(features[test])
    features = selection.transform(features)
    
   
    model.fit(features_trn , features_trn ,  batch_size=len(train) ,  epochs=100)
    
    rd_dim = []
    
    for item in features:
        lst = []
        lst.append(item)
        lt= numpy.asarray(lst , dtype=float)
        pred = encoder.predict(lt)
        prediction = pred[0].tolist()
        rd_dim.append(prediction)
    
    
    rd_dim = numpy.asarray(rd_dim).astype(numpy.float32)
    
    clf  = svm.SVC(kernel='linear')
    clf.fit(rd_dim[train] ,  labels[train])
    result = clf.predict(rd_dim[test])
    score  = accuracy_score(result , labels[test])
    scores.append(score)

print(scores)
average_scores = round(sum(scores)/K , 3)
print(average_scores)









