import numpy

from preprocess import labels_preprocess , features_preprocess ,  features_test_preprocess

features1 = features_preprocess()
features2 = features_test_preprocess()
features = numpy.concatenate( (features1 ,features2) )

labels = labels_preprocess()
k = []

for lbl in labels:
   if lbl == 'AML':
       k.append([0])
   if lbl =='ALL':
       k.append([1])

lbls = numpy.asarray(k)
lbls = lbls.astype(numpy.float32)
prediction =[]
#print(labels)
#print(k)

for sublist in lbls:
    for item in sublist:
        prediction.append(item)


print()
#print(lbls)
print(labels)

labels_train = labels[38:]
print(labels_train)


#print(features)
