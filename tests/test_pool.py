import numpy as np
from multiprocessing import Pool

features = np.random.random((10,5))

def func(feature):
    return feature[0]

pool = Pool()
rst = pool.map(func,features)
pool.close()

print rst
print features