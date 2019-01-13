#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import glob


AVAILABLE_DATASET_LIST = ['jumpingmean', 'scalingvariance', 'gmm_v2', 'highdim',
                          'beedance', 'fishkiller', 'hasc', 'yahoo']


# dataset
parser = argparse.ArgumentParser(description='interface of running experiments for MMD baselines')
parser.add_argument('--dataroot', type=str, required=True, help='[ ./data/simulation | ./data ] prefix path to data directory')
parser.add_argument('--dataset', type=str, default='beedance', help='dataset name ')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

# hyperparameters for grid search
parser.add_argument('--wnd_dim_list', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help='list of window dimensions')
parser.add_argument('--lambda_ae', type=float, default=1e-3, help='list of lambda_ae, coefficient of reconstruction loss')
parser.add_argument('--lambda_real', type=float, default=1, help='list of lambda_real, coefficient of MMD2(X_p_enc, X_f_enc)')
parser.add_argument('--max_iter', type=int, default=2000, help='max iteration for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size for training')
parser.add_argument('--eval_freq', type=int, default=25, help='evaluation frequency per batch updates')
parser.add_argument('--weight_clip', type=float, default=.1, help='weight clipping for netD')

# experiment log
parser.add_argument('--save_dir', type=str, default='experiment_log', help='experiment directory for saving train log and models')

# sanity check
args = parser.parse_args()
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
assert(os.path.isdir(args.dataroot))
assert(args.dataset in AVAILABLE_DATASET_LIST)
assert(os.path.isdir(args.save_dir))

dataroot = args.dataroot
dataset = args.dataset
gpu = args.gpu
wnd_dim_list = args.wnd_dim_list
lambda_ae = args.lambda_ae
lambda_real = args.lambda_real
max_iter = args.max_iter
batch_size = args.batch_size
eval_freq = args.eval_freq
weight_clip = args.weight_clip


def run_klcpd_exp(dataset):
    for wnd_dim in wnd_dim_list:
        data_dir = os.path.join(dataroot, dataset)
        for data_path in glob.glob('%s/*.mat' % (data_dir)):
            data_name = data_path.split('/')[-1].split('.')[0]
            save_name = '%s.wnd-%d.lambda_ae-%f.lambda_real-%f.clip-%f' % (data_name, wnd_dim, lambda_ae, lambda_real, weight_clip)
            save_path = '%s/%s' % (args.save_dir, save_name)
            trn_log_path = '%s.trn.log' % (save_path)
            option = '--data_path %s --wnd_dim %d --lambda_ae %f --lambda_real %f --weight_clip %f --max_iter %d --batch_size %d --eval_freq %d --save_path %s' \
                % (data_path, wnd_dim, lambda_ae, lambda_real, weight_clip, max_iter, batch_size, eval_freq, save_path)
            cmd = 'CUDA_VISIBLE_DEVICES=%d python -u klcpd.py %s 2>&1 | tee %s' % (gpu, option, trn_log_path)
            #print(cmd)
            os.system(cmd)


# main function call
run_klcpd_exp(dataset)
