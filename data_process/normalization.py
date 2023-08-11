import numpy as np
import torch

class Standard(object):

    def __init__(self):
        pass

    def fit(self, X):
        self.std = np.std(X)
        self.mean = np.mean(X)
        # if len(X.shape) == 2:
        #     self.multi_attr_flag = False
        #     self.std = np.std(X)
        #     self.mean = np.mean(X)
        # elif len(X.shape) == 3: 
        #     self.multi_attr_flag = True 
        #     X = X.reshape(-1, X.shape[-1])
        #     self.std = np.std(X, axis=0)
        #     self.mean = np.mean(X, axis=0)
        # else:
        #     assert False, 'current normalization does not support this data.'
        print("std:", self.std, self.std.shape, "mean:", self.mean, self.mean.shape)

    def transform(self, X):
        X = 1. * (X - self.mean) / self.std
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = X * self.std + self.mean
        # if isinstance(X, torch.Tensor) and self.multi_attr_flag:
        #     std = torch.FloatTensor(self.std).to(X.device)
        #     mean = torch.FloatTensor(self.mean).to(X.device)
        # else:
        #     print(f'x: {type(X)}')
        #     std = self.std
        #     mean = self.mean
        # X = X * std + mean
        return X

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

    # def rmse_transform(self, X):
    #     X = X * self.std
    #     return X
    # def mae_transform(self, X):
    #     X = X* self.std
    #     return X