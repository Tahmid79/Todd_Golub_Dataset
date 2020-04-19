import tensorflow as tf
from tensorflow import keras
import math
import numpy
from sklearn.metrics import  accuracy_score
from sklearn.decomposition import PCA

from data.preprocess import features_preprocess , features_test_preprocess , labels_preprocess

model = keras.Sequential()

input_layer = keras.layers.Dense(5 , input_shape=[15] , activation = 'tanh')
model.add(input_layer)

hidden_layer = keras.layers.Dense(5 , activation = 'tanh')
model.add(hidden_layer)

hidden_layer2 = keras.layers.Dense( 8  , activation = 'relu')
model.add(hidden_layer2)

#hidden_layer3 = keras.layers.Dense(4 ,  activation = 'relu')
#model.add(hidden_layer3)

output_layer = keras.layers.Dense(1, activation = 'sigmoid')
model.add(output_layer)

#gd = tf.train.GradientDescentOptimizer(0.01)
gd= tf.optimizers.SGD(learning_rate = 0.01 , name = 'SGD')

model.compile(optimizer=gd , loss='mean_squared_error' )

#input_data = [[1,1,0],[1,1,1],[0,1,0],[-1,1,0],[-1,0,0],[-1,0,1],[0,0,1],[1,1,0],[1,0,0],[-1,0,0],[1,0,1],[0,1,1],[0,0,0],[-1,1,1]]
#label_data = [[0],[0],[1],[1],[1],[0],[1],[0],[1],[1],[1],[1],[1],[0]]

labels = labels_preprocess()
k = []

for lbl in labels:
   if lbl == 'AML':
       k.append([0])
   if lbl =='ALL':
       k.append([1])


lbls = numpy.asarray(k)
lbls = lbls.astype(numpy.float32)

ftrs = numpy.concatenate( (features_preprocess() ,features_test_preprocess()) )
pca = PCA(n_components=15)
ftrs = pca.fit_transform(ftrs)


inp = numpy.asarray(ftrs)
inp = inp.astype(numpy.float32 )

val = numpy.asarray(lbls)
val = val.astype(numpy.float32)

x = tf.Variable(inp)
y = tf.Variable(val)

model.fit(x, y , epochs=500 , steps_per_epoch=15 )
results =  model.predict(inp , verbose=0 , steps=1)
prediction = []
pred_lbl = []

for sublist in results:
    for item in sublist:
        prediction.append(item)

for lbl in prediction:
    if lbl>=0.5:
     pred_lbl.append('ALL')
    if lbl < 0.5:
        pred_lbl.append('AML')



accuracy = accuracy_score(pred_lbl , labels )

print(results)
#print(prediction)
#print(pred_lbl)
print("Accuracy = " ,  accuracy*100 )

a = numpy.average(tf.metrics.MSE(y, results ))
print(a)



