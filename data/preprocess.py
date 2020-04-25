import pandas
import numpy
import os , sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def labels_preprocess():
    labels = pandas.read_csv('../data/actual.csv', index_col=0)
    #labels_train = labels.iloc[:38]
    lbl_series = labels['cancer']
    lbl = lbl_series.tolist()
    return lbl

def labels_preprocess_num():
    labels = labels_preprocess()
    k = []
    for item in labels:
        if item == 'AML':
            k.append([0])
        if item == 'ALL':
            k.append([1])

    k = numpy.asarray(k)
    return k



def features_preprocess():
    train = pandas.read_csv('../data/data_set_ALL_AML_train.csv')
    train = train[[col for col in train.columns if col.startswith('call') == False]]
    train.set_index('Gene Accession Number')
    #print(train.columns)

    cl = []
    sorted_df = pandas.DataFrame()

    for col in train.columns[2:]:
        col = int(col)
        cl.append(col)

    #print()
    cl.sort()

    ft = []

    for pt_no in cl:
        frame = train.loc[::, str(pt_no)]
        sorted_df[str(pt_no)] = frame

        values = frame.values
        values = list(values)
        ft.append(values)

    sorted_df = sorted_df.set_index(train['Gene Accession Number'])

    features = numpy.array(ft, dtype=float)
    return features

def features_test_preprocess():
    train = pandas.read_csv('../data/data_set_ALL_AML_independent.csv')
    train = train[[col for col in train.columns if col.startswith('call') == False ]]
    train.set_index('Gene Accession Number')
    #print(train.columns)

    cl = []
    sorted_df = pandas.DataFrame()

    for col in train.columns[2:]:
        col = int(col)
        cl.append(col)

    #print()
    cl.sort()

    ft = []

    for pt_no in cl:
        frame = train.loc[:: , str(pt_no)  ]
        sorted_df[str(pt_no)] = frame

        values = frame.values
        values = list(values)
        ft.append(values)

    sorted_df = sorted_df.set_index(train['Gene Accession Number'])

    features = numpy.array(ft, dtype=float)
    return features

def ft_lbls_num():
    features = numpy.concatenate((features_preprocess(), features_test_preprocess()))
    labels = labels_preprocess_num()
    return (features , labels)


