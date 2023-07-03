#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy.io as sio
import math
import torch
from torch.utils.data import Dataset


class HankelDataset(Dataset):
    def __init__(self, ts: np.array, p_wnd_dim:int=25, f_wnd_dim:int=10, sub_dim:int=1):
        """
        @param ts - timeseries
        @param p_wnd_dim - past window size
        @param f_wnd_dim - future window size
        @param sub_dim - Hankel matrix size
        """
        super().__init__()
        self.p_wnd_dim = p_wnd_dim
        self.f_wnd_dim = f_wnd_dim
        self.sub_dim = sub_dim

        if len(ts.shape) == 1: 
            ts = np.expand_dims(ts, -1)
        self.Y = ts # Timeseries
        self.T, self.D = ts.shape # T: time length; D: variable dimension
        self.var_dim = self.D * self.sub_dim

        self.Y_hankel = self.ts_to_hankel(self.Y)

    # prepare subspace data (Hankel matrix)
    def ts_to_hankel(self, ts):
        # T x D x sub_dim
        Y_hankel = np.zeros((self.T, self.D, self.sub_dim))
        for t in range(self.sub_dim, self.T):
            for d in range(self.D):
                Y_hankel[t, d, :] = ts[t-self.sub_dim+1:t+1, d].flatten()

        # Y_hankel is now T x (Dxsub_dim)
        Y_hankel = Y_hankel.reshape(self.T, -1)
        return Y_hankel


    # convert augmented data in Hankel matrix to origin time series
    # input: X_f, whose shape is batch_size x seq_len x (D*sub_dim)
    # output: Y_t, whose shape is batch_size x D
    def hankel_to_ts(self, X_f):
        batch_size = X_f.shape[0]
        Y_t = X_f[:, 0, :].contiguous().view(batch_size, self.D, self.sub_dim)
        return Y_t[:, :, -1]

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        if idx < 0: 
            idx += len(self)
        assert(0 <= idx < len(self))
        data = np.concatenate([
            np.tile(self.Y_hankel[:1,:], (self.p_wnd_dim, 1)), # left padding
            self.Y_hankel[:, :], # timeseries
            np.tile(self.Y_hankel[-1:,:], (self.f_wnd_dim, 1)), # right padding
            ])
        return {
            'X_p': torch.from_numpy(data[idx:idx+self.p_wnd_dim, :]),
            'X_f': torch.from_numpy(data[idx+self.p_wnd_dim:idx+self.p_wnd_dim+self.f_wnd_dim, :]),
            'Y': torch.from_numpy(self.Y[min(max(idx, 0), self.T-1)])
        }
