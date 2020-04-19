import pandas
import numpy

import sys
sys.path.append('./data/')
sys.path.append('../data/')



def preprocess_ft_lbls():

    dataframe = pandas.read_csv('C:\Jetbrains\PyCharm Projects\Todd_Golub_Dataset\data\expression_data.txt' , index_col=0  , sep='\t')

    patient = dataframe[[col for col in dataframe.columns if col.startswith('MLL')==False ]]

    labels = []
    features = []


    for clm in patient.columns[1:]:
        lbl =''
        crt_pt = patient.loc[:: , clm] # the current patient
        fts = crt_pt.values  # the gene expression for current patient

        if clm.startswith('AML')==True:   # recording the label for the current patient
            lbl = 'AML'
        elif clm.startswith('ALL')==True:
            lbl = 'ALL'

        labels.append(lbl)
        features.append(fts)



    features = numpy.asarray(features ,  dtype=numpy.float32)
    print(labels)
    print(type(features))

    return (features , labels)

def preprocess_ft_lbls_num():

    dataframe = pandas.read_csv('../data/expression_data.txt' , index_col=0  , sep='\t')
    #dataframe = pandas.read_csv('C:\Jetbrains\PyCharm Projects\Todd_Golub_Dataset\data\expression_data.txt' , index_col=0  , sep='\t')

    patient = dataframe[[col for col in dataframe.columns if col.startswith('MLL')==False ]]

    labels = []
    features = []


    for clm in patient.columns[1:]:
        lbl =''
        crt_pt = patient.loc[:: , clm] # the current patient
        fts = crt_pt.values  # the gene expression for current patient

        if clm.startswith('AML')==True:   # recording the label for the current patient
            lbl = [0]
        elif clm.startswith('ALL')==True:
            lbl = [1]

        labels.append(lbl)
        features.append(fts)



    features = numpy.asarray(features ,  dtype=numpy.float32)
    labels = numpy.asarray(labels ,  dtype=numpy.float32)

    return (features , labels)










