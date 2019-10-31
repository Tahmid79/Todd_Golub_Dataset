import pandas as pd
import numpy

labels = pd.read_csv('actual.csv' , index_col=0)
#print(labels)

train = pd.read_csv('data_set_ALL_AML_train.csv')

train = train[[col for col in train.columns if col.startswith('call')==False]]
train.set_index('Gene Accession Number')
print(train.columns)

"""
patient_1 = train['1']
pt1 = patient_1.values
pt1 = pt1.astype(float)

print( patient_1.head() )
print( pt1 )

print(len(train.columns))
"""

cl = []
sorted_df = pd.DataFrame()

for col in train.columns[2:]:
    col = int(col)
    #print(col, end=' ')
    cl.append(col)

print()
cl.sort()

print(cl)

ft = []

for pt_no in cl:
    frame = train.loc[::, str(pt_no)]
    sorted_df[str(pt_no)] = frame


    values = frame.values
    #values = list(values.astype(float) )
    values = list(values)
    ft.append(values)



sorted_df = sorted_df.set_index(train['Gene Accession Number'])

features = numpy.array(ft, dtype=float)

print( features )


"""
patient_1 = train.loc[::, '1']
print( patient_1.head() )
"""