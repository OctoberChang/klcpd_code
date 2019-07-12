# KL-CPD Pytorch Implementation
Code accompanying the ICLR 2019 paper [Kernel Change-point Detection with Auxiliary Deep Generative Models](https://openreview.net/forum?id=r1GbfhRqF7).


# Prerequisites
    - Python (v2.7)
    - PyTorch (v0.2.20)
    - scikit-learn

see 
```
  $ cat klcpd_py2.7_pt0.2.0_conda.txt
```
for an example of the detailed package dependencies configurations.


# Main Usage
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

# Quick Start on BeeDance dataset
For a quick start and experiment grid search, please execute ```run_klcpd.py```. For an example on BeeDance dataset:
```
    $ python run_klcpd.py --dataroot ./data --dataset beedance --wnd_dim_list 25 --max_iter 2000 --batch_size 64 
```

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
@article{chang2019kernel,
  title={Kernel change-point detection with auxiliary deep generative models},
  author={Chang, Wei-Cheng and Li, Chun-Liang and Yang, Yiming and P{\'o}czos, Barnab{\'a}s},
  journal={arXiv preprint arXiv:1901.06077},
  year={2019}
}
```

For any questions and comments, please send your email to
[wchang2@cs.cmu.edu](mailto:wchang2@cs.cmu.edu)


