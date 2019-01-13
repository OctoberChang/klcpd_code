#!/usr/bin/env python
# encoding: utf-8


import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]


# X_p_enc: batch_size x seq_len x hid_dim
# X_f_enc: batch_size x seq_len x hid_dim
# hid_dim could be either dataspace_dim or codespace_dim
# return: MMD2(X_p_enc[i,:,:], X_f_enc[i,:,:]) for i = 1:batch_size
def batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var):
    # some constants
    n_basis = 1024
    gumbel_lmd = 1e+6
    cnst = math.sqrt(1. / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, seq_len, nz = X_p_enc.size()

    # gumbel trick to get masking matrix to uniformly sample sigma
    # input: (batch_size*n_samples, nz)
    # output: (batch_size, n_samples, nz)
    def sample_gmm(W, batch_size):
        U = torch.cuda.FloatTensor(batch_size*n_samples, n_mixtures).uniform_()
        sigma_samples = F.softmax(U * gumbel_lmd).matmul(sigma_var)
        W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
        W_gmm = W_gmm.view(batch_size, n_samples, nz)
        return W_gmm

    W = Variable(torch.cuda.FloatTensor(batch_size*n_samples, nz).normal_(0, 1))
    W_gmm = sample_gmm(W, batch_size)                                   # batch_size x n_samples x nz
    W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()                   # batch_size x nz x n_samples
    XW_p = torch.bmm(X_p_enc, W_gmm)                                    # batch_size x seq_len x n_samples
    XW_f = torch.bmm(X_f_enc, W_gmm)                                    # batch_size x seq_len x n_samples
    z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
    z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
    batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1))**2, 1)
    return batch_mmd2_rff


