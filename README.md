# KL-CPD Pytorch Implementation
Code accompanying the ICLR 2019 paper [Kernel Change-point Detection with Auxiliary Deep Generative Models](https://openreview.net/forum?id=r1GbfhRqF7).


# Prerequisites
    - Python (v2.7)
    - PyTorch (v0.2.20)

see 
```
  $ cat klcpd_py2.7_pt0.2.0_conda.txt
```
for a example of the detailed package dependencies configurations.


# Usage
```
python klcpd.py [OPTIONS]
OPTIONS:
    --data_path DATA_PATH         data path to dataset.mat
    --trn_ratio TRN_RATIO         how much data used for training
    --val_ratio VAL_RATIO         how much data used for validation
    --gpu GPU                     gpu device id
    --cuda CUDA                   use gpu or not
    --random_seed RANDOM_SEED     random seed
    --wnd_dim WND_DIM             window size (past and future)
    --sub_dim SUB_DIM             dimension of subspace embedding
    --RNN_hid_dim RNN_HID_DIM     number of RNN hidden units
    --batch_size BATCH_SIZE       batch size for training
    --max_iter MAX_ITER           max iteration for pretraining RNN
    --optim OPTIM                 sgd|rmsprop|adam for optimization method
    --lr LR                       learning rate
    --weight_decay WEIGHT_DECAY   weight decay (L2 regularization)
    --momentum MOMENTUM           momentum for sgd
    --grad_clip GRAD_CLIP         gradient clipping for RNN (both netG and netD)
    --eval_freq EVAL_FREQ         evaluation frequency per generator update
    --CRITIC_ITERS CRITIC_ITERS   number of updates for critic per generator
    --weight_clip WEIGHT_CLIP     weight clipping for crtic
    --lambda_ae LAMBDA_AE         coefficient for the reconstruction loss
    --lambda_real LAMBDA_REAL     coefficient for the real MMD2 loss
    --save_path SAVE_PATH         path to save the final model
    --save_name SAVE_NAME         model/prediction names   
```

For a quick start, please execute ```run_klcpd.py```. See, for example
```
    $ python  [mnist/cifar10/celeba/lsun]
```

# Dataset
For mnist and cifar10, the dataset will be automatically download if not exist in
the designated DATAROOT directory.

For CelebA and LSUN dataset, please run the download script in ./data directory.


# More Info
This repository is by
[Wei-Cheng Chang](https://octoberchang.github.io/),
[Chun-Liang Li](http://www.cs.cmu.edu/~chunlial/),
[Yiming Yang](http://www.cs.cmu.edu/~yiming/),
[Barnabás Póczos](http://www.cs.cmu.edu/~bapoczos/),
and contains the source code to
reproduce the experiments in our paper
[Kernel Change-point Detection with Auxiliary Deep Generative Models](https://openreview.net/forum?id=r1GbfhRqF7).
If you find this repository helpful in your publications, please consider citing our paper.
```
@article{chang2018kernel,
  title={Kernel Change-point Detection with Auxiliary Deep Generative Models},
  author={Chang, Wei-Cheng and Li, Chun-Liang and Yang, Yiming and P{\'o}czos, Barnab{\'a}s},
  year={2018}
}
```

For any questions and comments, please send your email to
[wchang2@cs.cmu.edu](mailto:wchang2@cs.cmu.edu)


