#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import numpy as np
from data_loader import DataLoader
from sklearn.metrics.pairwise import euclidean_distances

def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='path to data in matlab format')
parser.add_argument('--trn_ratio', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--val_ratio', type=float, default=0.8,help='how much data used for validation')
parser.add_argument('--wnd_dim', type=int, required=True, default=10, help='window size (past and future)')
parser.add_argument('--sub_dim', type=int, default=1, help='dimension of subspace embedding')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
args = parser.parse_args()
print(args)

Data = DataLoader(args, trn_ratio=args.trn_ratio, val_ratio=args.val_ratio)
median_list = median_heuristic(Data.Y_subspace, beta=.5)
print(median_list)
