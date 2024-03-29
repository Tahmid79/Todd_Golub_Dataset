import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle
from mpl_toolkits import  mplot3d
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap ,SpectralEmbedding
import matplotlib


from data.preprocess_2nd import  preprocess_ft_lbls_num
from data.preprocess import  ft_lbls_num


matplotlib.use('TkAgg')

#embedding = PCA(n_components=3)

#embedding = LocallyLinearEmbedding(n_components=3 )
#embedding = Isomap( n_components=3)
embedding = MDS(n_components=3)



(features1 ,  labels1) = ft_lbls_num()
(features2 ,  labels2) = preprocess_ft_lbls_num()


#selection  =SelectKBest(k=1000)
#features1 = selection.fit_transform(features1, labels1)
#features2 = selection.fit_transform(features2, labels2)


features1 = embedding.fit_transform(features1, labels1)
features2 = embedding.fit_transform(features2, labels2)

fig = pyplot.figure()
ax = pyplot.axes(projection='3d')

#pyplot.xlabel('X-Axis' ,  fontsize='large')
#pyplot.ylabel('Y-Axis',  fontsize='large')
pyplot.title('Isomap')


features = features1.tolist()
labels =   [item[0]  for item in labels1.tolist()]

colors = ['red' if l==0  else 'blue'  for l in labels ]


x_points = [point[0] for point in features]
y_points = [point[1] for point in features]
z_points = [point[2] for point in features]

ax.scatter3D(x_points , y_points , z_points , cmap='hsv', color=colors  )



#//////////////////////////////////////////////


features = features2.tolist()
labels =   [item[0]  for item in labels2.tolist()]

colors = ['darkred' if l==0  else 'darkblue'  for l in labels ]


x_points = [point[0] for point in features]
y_points = [point[1] for point in features]
z_points = [point[2] for point in features]

ax.scatter3D(x_points , y_points , z_points , cmap='hsv' ,  color=colors  )


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























