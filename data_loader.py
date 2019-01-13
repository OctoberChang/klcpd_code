#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy.io as sio
import math
import torch
from torch.autograd import Variable


class DataLoader(object):
    def __init__(self, args, trn_ratio=0.6, val_ratio=0.8):
        self.cuda = args.cuda
        self.data_path = args.data_path
        self.p_wnd_dim = 25
        self.f_wnd_dim = args.wnd_dim
        self.sub_dim = args.sub_dim
        self.batch_size = args.batch_size

        # load data
        self.load_data(trn_ratio=trn_ratio, val_ratio=val_ratio)

        # prepare data
        self.prepare_data()

        # split data into trn/val/tst set
        self.split_data()

    # load data
    def load_data(self, trn_ratio=0.6, val_ratio=0.8):
        assert(os.path.lexists(self.data_path))
        dataset = sio.loadmat(self.data_path)
        self.Y = dataset['Y']                                   # Y: time series data, time length x number of variables
        self.L = dataset['L']                                   # L: label of anomaly, time length x 1
        self.T, self.D = self.Y.shape                           # T: time length; D: variable dimension
        self.n_trn = int(np.ceil(self.T * trn_ratio))           # n_trn: first index of val set
        self.n_val = int(np.ceil(self.T * val_ratio))           # n_val: first index of tst set
        self.var_dim = self.D * self.sub_dim

    # prepare subspace data (Hankel matrix)
    def prepare_data(self):
        # T x D x sub_dim
        self.Y_subspace = np.zeros((self.T, self.D, self.sub_dim))
        for t in range(self.sub_dim, self.T):
            for d in range(self.D):
                self.Y_subspace[t, d, :] = self.Y[t-self.sub_dim+1:t+1, d].flatten()

        # Y_subspace is now T x (Dxsub_dim)
        self.Y_subspace = self.Y_subspace.reshape(self.T, -1)

    # split data into trn/val/tst set
    def split_data(self):
        trn_set_idx = range(self.p_wnd_dim, self.n_trn)
        val_set_idx = range(self.n_trn, self.n_val)
        tst_set_idx = range(self.n_val, self.T)
        print('n_trn ', len(trn_set_idx), 'n_val ', len(val_set_idx), 'n_tst ', len(tst_set_idx))
        self.trn_set = self.__batchify(trn_set_idx)
        self.val_set = self.__batchify(val_set_idx)
        self.tst_set = self.__batchify(tst_set_idx)

    # convert augmented data in Hankel matrix to origin time series
    # input: X_f, whose shape is batch_size x seq_len x (D*sub_dim)
    # output: Y_t, whose shape is batch_size x D
    def repack_data(self, X_f, batch_size):
        Y_t = X_f[:, 0, :].contiguous().view(batch_size, self.D, self.sub_dim)
        return Y_t[:, :, -1]

    def __batchify(self, idx_set):
        n = len(idx_set)
        L = torch.zeros((n, 1))                             # anomaly label
        Y = torch.zeros((n, self.D))                        # true signal
        X_p = torch.zeros((n, self.p_wnd_dim, self.var_dim))  # past window buffer
        X_f = torch.zeros((n, self.f_wnd_dim, self.var_dim))  # future window buffer

        # XXX: dirty trick to augment the last buffer
        data = np.concatenate((self.Y_subspace, self.Y_subspace[-self.f_wnd_dim:, :]))
        for i in range(n):
            l = idx_set[i] - self.p_wnd_dim
            m = idx_set[i]
            u = idx_set[i] + self.f_wnd_dim
            X_p[i, :, :] = torch.from_numpy(data[l:m, :])
            X_f[i, :, :] = torch.from_numpy(data[m:u, :])
            Y[i, :] = torch.from_numpy(self.Y[m, :])
            L[i] = torch.from_numpy(self.L[m])
        return {'X_p': X_p, 'X_f': X_f, 'Y': Y, 'L': L}

    def get_batches(self, data_set, batch_size, shuffle=False):
        X_p, X_f = data_set['X_p'], data_set['X_f']
        Y, L = data_set['Y'], data_set['L']
        length = len(Y)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        s_idx = 0
        while (s_idx < length):
            e_idx = min(length, s_idx + batch_size)
            excerpt = index[s_idx:e_idx]
            X_p_batch, X_f_batch = X_p[excerpt], X_f[excerpt]
            Y_batch, L_batch = Y[excerpt], L[excerpt]
            if self.cuda:
                X_p_batch = X_p_batch.cuda()
                X_f_batch = X_f_batch.cuda()
                Y_batch = Y_batch.cuda()
                L_batch = L_batch.cuda()

            data = [Variable(X_p_batch),
                    Variable(X_f_batch),
                    Variable(Y_batch),
                    Variable(L_batch)]
            yield data
            s_idx += batch_size

