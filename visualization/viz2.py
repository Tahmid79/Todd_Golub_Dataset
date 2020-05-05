import pandas
import numpy
import matplotlib.patches as mpatches
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle
from mpl_toolkits import  mplot3d
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap ,SpectralEmbedding
import matplotlib
import keras
from keras.models import Model
from keras.layers import Input ,  Dense ,  LeakyReLU


from data.preprocess import  ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num

selection = SelectKBest(k=50)


(features1 , labels1) = ft_lbls_num()
features1 = selection.fit_transform(features1 , labels1)


(features2, labels2) = preprocess_ft_lbls_num()
features2 = selection.fit_transform(features2 , labels2)


input = Input(shape=(50, ))

hd =  LeakyReLU()( Dense(3)(input)  )

output =  Dense(50 ,  activation='softmax')(hd)

model = Model( input , output)
encoder= Model( input, hd )

model.compile(loss='mse', optimizer='sgd')



for i in range(300):

    if ( i%2==0):
        model.fit(features1 ,  features1 ,  batch_size=len(features1) , epochs=20 , validation_split=0.8 )

    if (i%2==1):
        model.fit(features2, features2, batch_size=len(features2), epochs=20 ,  validation_split=0.8  )



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


features1 = rd_dim
features2 = rd_dim2

matplotlib.use('TkAgg')


fig = pyplot.figure()
ax = pyplot.axes(projection='3d')
pyplot.title('Autoencoder')

#//////////////////////////////////////////////


features = features1.tolist()
labels =   [item[0]  for item in labels1.tolist()]

colors = ['red' if l==0  else 'blue'  for l in labels ]


x_points = [point[0] for point in features]
y_points = [point[1] for point in features]
z_points = [point[2] for point in features]

ax.scatter3D(x_points , y_points , z_points , cmap='hsv', color=colors  )



#//////////////////////////////////////////////

"""
features = features2.tolist()
labels =   [item[0]  for item in labels2.tolist()]

colors = ['darkred' if l==0  else 'darkblue'  for l in labels ]


x_points = [point[0] for point in features]
y_points = [point[1] for point in features]
z_points = [point[2] for point in features]

ax.scatter3D(x_points , y_points , z_points , cmap='hsv' ,  color=colors  )
"""

#//////////////////////////////////////////////


# build the legend
red_patch = mpatches.Patch(color='red', label='AML 1st')
blue_patch = mpatches.Patch(color='blue', label='ALL 1st')
dred_patch = mpatches.Patch(color='darkred', label='AML 2nd')
dblue_patch = mpatches.Patch(color='darkblue', label='ALL 2nd')

# set up for handles declaration
patches = [red_patch, blue_patch, dred_patch, dblue_patch]

# define and place the legend
#legend = ax.legend(handles=patches,loc='upper right')

# alternative declaration for placing legend outside of plot
legend = ax.legend(handles=patches,bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)



pyplot.show()