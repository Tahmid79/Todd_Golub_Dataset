import numpy
import sklearn

from ptc.dt1_prtc import preprocess_ft_lbls


(features , labels) =  preprocess_ft_lbls()

print(features.shape)
print(len(labels))


