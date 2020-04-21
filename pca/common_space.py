import pandas
import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import  pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
from sklearn.feature_selection import SelectKBest
from sklearn.utils import  shuffle

from data.preprocess import ft_lbls_num
from data.preprocess_2nd import preprocess_ft_lbls_num

(features , labels) = ft_lbls_num()

pca = PCA(n_components=10)

K= 5














