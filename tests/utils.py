import pandas as pd
import time
import gc

df = pd.read_csv('data/dataset1.csv')
samples = []
start = time.time()
gc.disable()
for i in range(df.shape[0]):
    samples.append(df.iloc[i])
gc.enable()
stop = time.time()
print stop - start
