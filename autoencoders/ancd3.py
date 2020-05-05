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
from sklearn.naive_bayes import GaussianNB

from data.preprocess import ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num

(features, labels) = ft_lbls_num()

selection = SelectKBest(k=50)
features = selection.fit_transform(features , labels)



input = Input(shape=(50,))
hd1 = Dense(20)(input)
leaky = LeakyReLU()(hd1)
output = Dense(50, activation='softmax')(hd1)


model = Model(input, output)
encoder = Model(input, hd1)




model.compile(optimizer='sgd' , loss='mean_squared_error')
model.fit(features , features , batch_size=len(features) ,  epochs=2000 , validation_split=0.9    )


rd_dim = []


for item in features:
    lst = []
    lst.append(item)
    lt = numpy.asarray(lst, dtype=float)
    pred = encoder.predict(lt)
    prediction = pred[0].tolist()
    rd_dim.append(prediction)

rd_dim = numpy.asarray(rd_dim).astype(numpy.float32)

features = rd_dim

K = 5
cv = KFold(n_splits=K, shuffle=True)
scores = []

clf = svm.LinearSVC()

for i in range(100):

    features , labels = shuffle(features, labels)

    for train, test in cv.split(features):
        features_trn = features[train]
        features_test = features[test]

        labels_trn = labels[train]
        labels_test = labels[test]


        clf.fit(features_trn, labels_trn)
        result = clf.predict(features_test)
        score = accuracy_score(result, labels_test)
        scores.append(score)





print()
print(scores)
print('Average Score = ' , round( sum(scores)/(len(scores)) ,  5  )  )
print('Standard Deviation = ' , numpy.std(  scores ) )












