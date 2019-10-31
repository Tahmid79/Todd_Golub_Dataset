import pandas
import numpy

def labels_preprocess():
    labels = pandas.read_csv('actual.csv', index_col=0)
    #labels_train = labels.iloc[:38]
    lbl_series = labels['cancer']
    lbl = lbl_series.tolist()
    return lbl

def features_preprocess():
    train = pandas.read_csv('data_set_ALL_AML_train.csv')
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
    train = pandas.read_csv('data_set_ALL_AML_independent.csv')
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
        frame = train.loc[::, str(pt_no)  ]
        sorted_df[str(pt_no)] = frame

        values = frame.values
        values = list(values)
        ft.append(values)

    sorted_df = sorted_df.set_index(train['Gene Accession Number'])

    features = numpy.array(ft, dtype=float)
    return features


