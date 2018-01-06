import random
import numpy as np


class RowSampler(object):
    def __init__(self, n, sampling_rate=0.8):
        self.row_mask = np.ones((n,), dtype=bool)
        inds = np.random.choice(n, int(n*(1-sampling_rate)), replace=False)
        self.row_mask[inds] = False

    def shuffle(self):
        np.random.shuffle(self.row_mask)


class ColumnSampler(object):
    def __init__(self, n, sampling_rate=0.8):
        self.col_index = range(n)
        self.n_selected = int(n*sampling_rate)
        self.col_selected = self.col_index[:self.n_selected]

    def shuffle(self):
        random.shuffle(self.col_index)
        self.col_selected = self.col_index[:self.n_selected]