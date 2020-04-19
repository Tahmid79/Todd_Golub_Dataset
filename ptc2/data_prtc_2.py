import pandas
import numpy

labels = pandas.read_csv('actual.csv' , index_col=0)
labels_train = labels.iloc[:38]

lbl_series = labels_train['cancer']


lbl = lbl_series.tolist()
print(lbl)
print(len(lbl))



