from tgboost.bining import BinStructure
import numpy as np
import sys
import pandas as pd

"""
features = np.random.random((1000000, 50))
feature_dim = features.shape[1]
print features

bs = BinStructure(features, max_bin=10, estimation_sampling=1000000)
print "size of bs:", sys.getsizeof(bs)
for i in range(feature_dim):
    print bs[i]

bs1 = BinStructure(features, max_bin=10, estimation_sampling=100000)
print "size of bs1:", sys.getsizeof(bs1)
for i in range(feature_dim):
    print bs1[i]
"""

train = pd.read_csv("../../train.csv").values
bs2 = BinStructure(train)
for i in range(train.shape[1]):
    print i,"==============="
    print bs2[i]

