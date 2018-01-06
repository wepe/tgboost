# features: X, label: y, both are numpy array
# we first pre-sort each feature, then cut each feature into max_bin bins with equal distinct value
# in our implement, uint8 is used to represent the feature value after bining.
# so max value of max_bin is 256, it is good enough in practice
# design a BinStructure, which maintains bin information for each feature


import numpy as np


class BinStructure(object):
    def __init__(self,
                 features,
                 max_bin=256,
                 estimation_sampling=None):
        """
        :param features: it is numpy array, m rows n columns
        :param max_bin: the max number of the bins
        :param estimation_sampling: to estimate the bin bounder, first sampling data
        """
        assert len(features.shape) == 2
        assert 0 < max_bin <= 256
        self.features = features
        self.max_bin = max_bin

        if estimation_sampling is None:
            self.estimation_sampling = 100000
        else:
            self.estimation_sampling = estimation_sampling
        self.estimation_sampling = min(self.estimation_sampling, features.shape[0])

        self.feature_dim = features.shape[1]
        self.bins_upper_bounder = [{} for _ in range(self.feature_dim)]

        self.construct_bins_upper_bounder()

    def data_sampling(self, col):
        # sampling data from features[:,col], exclude those with nan value. return the selected index
        arr = ~np.isnan(self.features[:, col])
        arr = arr.nonzero()[0]
        size = min(arr.shape[0], self.estimation_sampling)
        return arr[np.random.choice(arr.shape[0], size, replace=False)]

    def construct_bins_upper_bounder(self):
        for i in range(self.feature_dim):
            distinct_value_cnt = {}
            selected_inds = self.data_sampling(i)
            for value in self.features[selected_inds, i]:
                if distinct_value_cnt.has_key(value):
                    distinct_value_cnt[value] += 1
                else:
                    distinct_value_cnt[value] = 1

            sorted_distinct_value = sorted(distinct_value_cnt.keys())

            if len(sorted_distinct_value) < self.max_bin:
                # number of distinct value is less than max_bin
                for j in range(len(sorted_distinct_value)):
                    self.bins_upper_bounder[i][j] = sorted_distinct_value[j]
            else:
                # number of distinct value is greater than max_bin
                # we should assign equal data into each bin,
                # and those with the same distinct value should in the same bin
                # we simply scan sorted_distinct_value, accumulate the cnt,
                # when greater than avg,  these distinct value can be put into one bin, so we get its upper bounder
                avg_cnt = int(len(selected_inds) / self.max_bin)
                acc_cnt = 0
                j = 0
                for distinct_value in sorted_distinct_value:
                    acc_cnt += distinct_value_cnt[distinct_value]
                    if acc_cnt >= avg_cnt:
                        self.bins_upper_bounder[i][j] = distinct_value
                        acc_cnt = 0
                        j += 1
                if acc_cnt != 0:
                    self.bins_upper_bounder[i][j] = sorted_distinct_value[-1]

        del self.features

    def __getitem__(self, item):
        return self.bins_upper_bounder[item]