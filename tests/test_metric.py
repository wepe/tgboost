import pandas as pd
from tgboost.metric import accuracy


# test accuracy
df = pd.read_csv('data/temp.csv')
preds = df.pred.round().values
labels = df.label.values
print accuracy(preds,labels)