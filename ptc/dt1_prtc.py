import pandas
import numpy

def preprocess_ft_lbls():

    dataframe = pandas.read_csv('expression_data.txt' , index_col=0  , sep='\t')

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

    dataframe = pandas.read_csv('expression_data.txt' , index_col=0  , sep='\t')

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










