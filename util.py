#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import sklearn.metrics
import cPickle as pickle


def load_data(data_path):
    data = sio.loadmat(data_path)
    return data['Y'], data['L']

def load_matlab_v1_log(data_path):
    eval_log = sio.loadmat(data_path)
    ret_dict = {'Y_tst': eval_log['Y_tst'],
                'L_tst': eval_log['L_tst'],
                'Y_tst_pred': eval_log['Y_tst_pred'],
                'L_tst_pred': eval_log['err'][:, 1], # use MSE as predict score
                'err': eval_log['err']}
    return ret_dict

def load_matlab_v2_log(data_path):
    eval_log = sio.loadmat(data_path)
    ret_dict = {'Y_tst': eval_log['Y_tst'],
                'L_tst': eval_log['L_tst'],
                'Y_tst_pred': None,
                'L_tst_pred': eval_log['Y_tst_pred'],
                'err': None}
    return ret_dict

def load_python_log(data_path):
    eval_log = pickle.load(open(data_path, 'rb'))
    ret_dict = {'Y_tst': eval_log['Y_true'],
                'L_tst': eval_log['L_true'],
                'Y_tst_pred': eval_log['Y_pred'],
                'L_tst_pred': eval_log['L_pred'],
                'err': None}
    return ret_dict

def compute_auc(eval_dict):
    L_true = eval_dict['L_tst'].flatten()
    L_pred = eval_dict['L_tst_pred'].flatten()
    # print('L_true', L_true.shape, 'L_pred', L_pred.shape)
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, L_pred)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    return fp_list, tp_list, auc

def compute_average_roc(tprs, base_fpr):
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    return mean_tprs

def forecast_loss(eval_dict):
    assert(eval_dict['Y_tst_pred'] is not None)
    sqr_err = np.sum((eval_dict['Y_tst'] - eval_dict['Y_tst_pred'])**2, axis=1)
    abs_err = np.sum(abs(eval_dict['Y_tst'] - eval_dict['Y_tst_pred']), axis=1)
    mse_mean = np.mean(sqr_err)
    mae_mean = np.mean(abs_err)
    return mse_mean, mae_mean

def print_auc_table(result_array, all_methods):
    # print auc for latex table
    print('metric', end='')
    for i, method in enumerate(all_methods):
        print(' & %s' % (method), end='')
    print('')
    print('AUC', end='')
    for i, method in enumerate(all_methods):
        print(' & %.4f' % (np.mean(result_array[i, :])), end='')
    print('')
