#!/usr/bin/env python
# encoding: utf-8

import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr=0.1, grad_clip=10.0, weight_decay=0., momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.max_norm = grad_clip
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.method = method
        self._makeOptimizer()

    def zero_grad(self):
        self.optimizer.zero_grad();

    def step(self):
        # Compute gradients norm.
        total_norm = 0
        for p in self.params:
            total_norm += p.grad.data.norm(2) ** 2
        total_norm = total_norm ** (1. / 2)
        clip_coef = self.max_norm / (total_norm + 1e-6)

        # grading clipping
        if clip_coef < 1:
            for p in self.params:
                p.grad.data.mul_(clip_coef)
        self.optimizer.step()

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl
        self._makeOptimizer()
