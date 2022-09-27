"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/ssd_detectors
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os, sys, time, warnings, itertools

from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.metrics import Mean

from .losses import *


class LearningRateDecay(Callback):
    def __init__(self, methode='linear', base_lr=1e-3, n_desired=40000, desired=0.1, bias=0.0, minimum=0.1):
        super(LearningRateDecay, self).__init__()
        self.methode = methode
        self.base_lr = base_lr
        self.n_desired = n_desired
        self.desired = desired
        self.bias = bias
        self.minimum = minimum
        
        #TODO: better naming

    def compute_learning_rate(self, n, methode):
        n_desired = self.n_desired
        desired = self.desired
        base_lr = self.base_lr
        bias = self.bias
        
        offset = base_lr * desired * bias
        base_lr = base_lr - offset
        
        desired = desired / (1-desired*bias) * (1-bias)
        
        if methode == 'default':
            k = (1 - desired) / n_desired
            lr = np.maximum( -k * n + 1, 0)
        elif methode == 'linear':
            k = (1 / desired - 1) / n_desired
            lr = 1 / (1 + k * n)
        elif methode == 'quadratic':
            k = (np.sqrt(1/desired)-1) / n_desired
            lr = 1 / (1 + k * n)**2
        elif methode == 'exponential':
            k = -1 * np.log(desired) / n_desired
            lr = np.exp(-k*n)
        
        lr = base_lr * lr + offset
        lr = np.maximum(lr, self.base_lr * self.minimum)
        return lr
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        steps_per_epoch = self.params['steps']
        iteration = self.epoch * steps_per_epoch + batch
        
        lr = self.compute_learning_rate(iteration, self.methode)
        K.set_value(self.model.optimizer.lr, lr)

    def plot_learning_rates(self):
        n = np.linspace(0, self.n_desired*2, 101)
        plt.figure(figsize=[16, 6])
        plt.plot([n[0], n[-1]], [self.base_lr*self.desired*self.bias]*2, 'k')
        for m in ['default', 'linear', 'quadratic', 'exponential']:
            plt.plot(n, self.compute_learning_rate(n, m))
        plt.legend(['bias', '$-kn+1$', '$1/(1+kn)$', '$1/(1+kn)^2$', '$e^{-kn}$'])
        plt.grid()
        plt.xlim(0, n[-1])
        plt.ylim(0, None)
        plt.show()


class ModelSnapshot(Callback):
    """Save the model weights after an interval of iterations."""
    
    def __init__(self, logdir, interval=10000, verbose=1):
        super(ModelSnapshot, self).__init__()
        self.logdir = logdir
        self.interval = interval
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch + 1
    
    def on_batch_end(self, batch, logs=None):
        if self.iteration % self.interval == 0:
            filepath = os.path.join(self.logdir, 'weights.%06i.h5' % (self.iteration))
            if self.verbose > 0:
                print('\nSaving model %s' % (filepath))
            self.model.save_weights(filepath, overwrite=True)


class Logger(Callback):
    
    def __init__(self, logdir):
        super(Logger, self).__init__()
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def save_history(self):
        df = pd.DataFrame.from_dict(self.model.history.history)
        df.to_csv(os.path.join(self.logdir, 'history.csv'), index=False)
    
    def append_log(self, logs):
        data = {k:[float(logs[k])] for k in self.model.metrics_names}
        data['iteration'] = [self.iteration]
        data['epoch'] = [self.epoch]
        data['batch'] = [self.batch]
        data['time'] = [time.time() - self.start_time]
        #data['lr'] = [float(K.get_value(self.model.optimizer.lr))]
        df = pd.DataFrame.from_dict(data)
        with open(os.path.join(self.logdir, 'log.csv'), 'a') as f:
            df.to_csv(f, header=f.tell()==0, index=False)
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.save_history()

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch
        
    def on_batch_end(self, batch, logs=None):
        self.append_log(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.save_history()


class MetricUtility():
    """History and Log for Tensorflow 2.x Eager API.
    
    # Arguments
        names: list of metric names.
        logdir: If specified, the log and history are written to csv files.
    
    The update methode receives a dictionary with values for each metric and
    should be called after each iteration.
    
    # Example
        mu = MetricUtility(['loss', 'accuracy'], logdir='./')
        for each epoch:
            mu.on_epoch_begin()
            for each training step:
                ...
                mu.update(metric_values, training=True)
            for each validation step:
                ...
                mu.update(metric_values, training=False)
            mu.on_epoch_end(verbose=True)
    """
    
    def __init__(self, names=['loss',], logdir=None, optimizer=None):
        self.names = names
        self.logdir = logdir
        self.optimizer = optimizer
        
        if logdir is not None:
            self.log_path = os.path.join(self.logdir, 'log.csv')
            self.history_path = os.path.join(self.logdir, 'history.csv')
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
            if os.path.exists(self.history_path):
                os.remove(self.history_path)
        
        self.reset()
        
    def reset(self):
        self.iteration = 0
        self.epoch = 0
        self.log = {n: [] for n in self.names}
        self.log.update({'epoch': [], 'time': []})
        self.history = {n: [] for n in self.names}
        self.history.update({'val_'+n: [] for n in self.names})
        self.history.update({'epoch': [], 'time': []})
        if self.optimizer is not None:
            self.log.update({'learning_rate': []})
            self.history.update({'learning_rate': []})
    
    def on_epoch_begin(self):
        if self.epoch == 0:
            self.t0 = time.time()
        self.t1 = time.time()
        self.epoch += 1
        self.steps = 0
        self.steps_val = 0
        self.metrics = {n: Mean() for n in self.names}
        self.metrics_val = {n: Mean() for n in self.names}
        
    def update(self, values, training=True):
        float_values = {n: float(v) for n, v in values.items()}
        if training:
            self.t2 = time.time()
            self.iteration += 1
            self.steps += 1
            for n, v in float_values.items():
                self.metrics[n].update_state(v)
                self.log[n].append(v)
            self.log['epoch'].append(self.epoch)
            self.log['time'].append(time.time()-self.t0)
            if self.optimizer is not None:
                self.log['learning_rate'].append(float(self.optimizer.learning_rate))
            
            if self.logdir is not None:
                float_values = {k:v[-1:] for k, v in self.log.items()}
                df = pd.DataFrame.from_dict(float_values)
                with open(self.log_path, 'a') as f:
                    df.to_csv(f, header=f.tell()==0, index=False)
        else:
            self.steps_val += 1
            for n, v in float_values.items():
                self.metrics_val[n].update_state(v)

    def on_epoch_end(self, verbose=True):
        if self.steps == 0:
            warnings.warn('no metric update was done')
            return
        
        float_values = {n: float(m.result()) for n, m in self.metrics.items()}
        if self.steps_val > 0:
            float_values.update({'val_'+n: float(m.result()) for n, m in self.metrics_val.items()})
        
        for n, v in float_values.items():
            self.history[n].append(v)
        self.history['epoch'].append(self.epoch)
        self.history['time'].append(time.time()-self.t0)
        if self.optimizer is not None:
            self.history['learning_rate'].append(float(self.optimizer.learning_rate))
        
        if self.logdir is not None:
            float_values = {k:v[-1:] for k, v in self.history.items() if len(v)}
            df = pd.DataFrame.from_dict(float_values)
            with open(self.history_path, 'a') as f:
                df.to_csv(f, header=f.tell()==0, index=False)
        
        if verbose:
            t1, t2, t3 = self.t1, self.t2, time.time()
            for n, v in self.history.items():
                if len(v):
                    print('%s %5.5f ' % (n, v[-1]), end='')
            print('\n%.1f minutes/epoch  %.2f iter/sec' % ((t3-t1)/60, self.steps/(t2-t1)))


def filter_signal(x, y, window_length=1000):
    if type(window_length) is not int or len(y) <= window_length:
        return [], []
    
    #w = np.ones(window_length) # moving average
    w = np.hanning(window_length) # hanning window
    
    wlh = int(window_length/2)
    if x is None:
        x = np.arange(wlh, len(y)-wlh+1)
    else:
        x = x[wlh:len(y)-wlh+1]
    y = np.convolve(w/w.sum(), y, mode='valid')
    return x, y


def plot_log(log_dirs, names=None, limits=None, window_length=250, filtered_only=False, autoscale=True, legend_loc='best'):
    """Plot and compares the training log contained in './checkpoints/'.
    
    # Agrumets
        log_dirs: string or list of string with directory names.
        names: list of strings with metric names in 'log.csv'.
            None means all.
        limits: tuple with min and max iteration that should be plotted.
            None means no limits.
        window_length: int, window length for signal filter.
            None means no filtered signal is plotted.
    
    # Notes
        The epoch is inferred from the log with the most iterations.
        Different batch size leads to different epoch length.
    """
    
    loss_terms = {'loss', 'error', 'abs', 'dist'}
    metric_terms = {'precision', 'recall', 'fmeasure', 'accuracy', 'sparsity', 'visibility'}
    
    if type(log_dirs) == str:
        log_dirs = [log_dirs]
    log_dirs = list(log_dirs)
    for d in [d for d in log_dirs]:
        if not os.path.isfile(os.path.join('.', 'checkpoints', d, 'log.csv')):
            print(d+' not found')
            log_dirs.remove(d)
    
    if limits is None:
        limits = slice(None)
    elif type(limits) in [list, tuple]:
        limits = slice(*limits)
    
    dfs = []
    max_df = []
    all_names = set()
    for d in log_dirs:
        df = pd.read_csv(os.path.join('.', 'checkpoints', d, 'log.csv'))
        all_names.update(df.keys())
        if len(df) > len(max_df):
            max_df = df
        if 'iteration' not in df.keys():
            df['iteration'] = np.arange(1,len(df)+1)
        if 'epoch' not in df.keys():
            df['epoch'] = np.zeros(len(df))
        df = df[limits]
        df = {k: np.array(df[k]) for k in df.keys()}
        dfs.append(df)
    
    iteration = max_df['iteration']
    epoch = max_df['epoch']
    idx = np.argwhere(np.diff(epoch))[:,0]
    
    if 'time' in max_df.keys() and len(idx) > 1:
        t = max_df['time']
        print('time per epoch %3.1f h' % ((t[idx[1]]-t[idx[0]])/3600))
    
    if names is None:
        print(all_names)
        names = all_names.difference({'time', 'epoch', 'iteration'})
    
    # reduce epoch ticks
    max_ticks = 20
    n = len(idx)
    if n > 1:
        n = round(n,-1*int(np.floor(np.log10(n))))
        while n >= max_ticks:
            if n/2 < max_ticks:
                n /= 2
            else:
                if n/5 < max_ticks:
                    n /= 5
                else:
                    n /= 10
        idx_step = int(np.ceil(len(idx)/n))
        epoch_step = epoch[idx[idx_step]] - epoch[idx[0]]
        for first_idx in range(len(idx)):
            if epoch[idx[first_idx]] % epoch_step == 0:
                break
        idx_red = [idx[i] for i in range(first_idx, len(idx), idx_step)]
    else:
        idx_red = idx
    
    colorgen = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = [next(colorgen) for i in range(len(dfs))]
    
    for k in names:
        plt.figure(figsize=(16, 6))
        xmin, xmax, ymax, ymean = 2147483647, 0, 0, 0
        non_zero = False
        for i, df in enumerate(dfs):
            if k in df.keys():
                if len(df[k]):
                    non_zero = True
                    if window_length:
                        plt.plot(*filter_signal(df['iteration'], df[k], window_length), color=colors[i], label=log_dirs[i])
                        if not filtered_only:
                            plt.plot(df['iteration'], df[k], zorder=0, color=colors[i], alpha=0.3)
                    else:
                        plt.plot(df['iteration'], df[k], zorder=0, color=colors[i], label=log_dirs[i])
                    xmin, xmax = min(xmin, df['iteration'][0]), max(xmax, df['iteration'][-1])
                    if np.all(np.isfinite(df[k])):
                        ymax = max(ymax, np.max(df[k]))
                        ymean = max(ymean, np.mean(df[k]))
                    else:
                        print(log_dirs[i]+' NaN or inf')
        
        ymax = min(ymax, ymean*4)
        
        if non_zero:
            plt.title(k, y=1.05)
            plt.legend(loc=legend_loc)
            
            ax1 = plt.gca()
            ax1.set_xlim(xmin, xmax)
            if autoscale:
                k_split = k.split('_')
                if len(loss_terms.intersection(k_split)):
                    plt.ylim(0, ymax)
                elif len(metric_terms.intersection(k_split)):
                    plt.ylim(0, 1)
            ax1.yaxis.grid(True)
            #ax1.set_xlabel('iteration')
            #ax1.set_yscale('linear')
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)
            
            ax2 = ax1.twiny()
            ax2.xaxis.grid(True)
            ax2.set_xticks(iteration[idx_red])
            ax2.set_xticklabels(epoch[idx_red])
            ax2.set_xlim(xmin, xmax)
            #ax2.set_xlabel('epoch')
            #ax2.set_yscale('linear')
            ax2.get_yaxis().get_major_formatter().set_useOffset(False)
            
            plt.show()
        else:
            #print(k+' no values')
            plt.close()

def plot_history(log_dirs, names=None, limits=None, autoscale=True, no_validation=False):

    loss_terms = {'loss', 'error', 'abs', 'dist'}
    metric_terms = {'precision', 'recall', 'fmeasure', 'accuracy', 'sparsity', 'visibility'}
    
    if type(log_dirs) == str:
        log_dirs = [log_dirs]
    log_dirs = list(log_dirs)
    for d in [d for d in log_dirs]:
        if not os.path.isfile(os.path.join('.', 'checkpoints', d, 'history.csv')):
            print(d+' not found')
            log_dirs.remove(d)
    
    if limits is None:
        limits = slice(None)
    elif type(limits) in [list, tuple]:
        limits = slice(*limits)
    
    dfs = []
    max_df = []
    all_names = set()
    for d in log_dirs:
        df = pd.read_csv(os.path.join('.', 'checkpoints', d, 'history.csv'))
        all_names.update(df.keys())
        if len(df) > len(max_df):
            max_df = df
        if 'epoch' not in df.keys():
            df['epoch'] = np.arange(1,len(df)+1)
        df = df[limits]
        df = {k: np.array(df[k]) for k in df.keys()}
        dfs.append(df)
    
    epoch = np.array(max_df['epoch'])
    
    if names is None:
        print(all_names)
        names = {n for n in all_names if not n.startswith('val_')}
        names = names.difference({'time', 'epoch'})
    
    colorgen = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = [next(colorgen) for i in range(len(dfs))]
    
    for k in names:
        plt.figure(figsize=(16,4))
        
        xmin, xmax = 2147483647, 0
        ymin, ymax = sys.float_info.max, sys.float_info.min
        for i, df in enumerate(dfs):
            if len(df['epoch']):
                xmin, xmax = min(xmin, df['epoch'][0]), max(xmax, df['epoch'][-1])
            if k in df.keys():
                plt.plot(df['epoch'], df[k], color=colors[i], label=log_dirs[i])
                if not len(df[k]):
                    pass
                elif np.all(np.isfinite(df[k])):
                    ymin, ymax = min(ymin, np.min(df[k])), max(ymax, np.max(df[k]))
                else:
                    print(log_dirs[i]+' NaN or inf')
            kv = 'val_'+k
            if not no_validation and kv in df.keys():
                plt.plot(df['epoch'], df[kv], '--', color=colors[i])
                if not len(df[kv]):
                    pass
                elif np.all(np.isfinite(df[kv])):
                    ymin, ymax = min(ymin, np.max(df[kv])), max(ymax, np.max(df[kv]))
                else:
                    print(log_dirs[i]+' NaN or inf')
        
        if ymax > sys.float_info.min:
            plt.xlim(xmin, xmax)
            if autoscale:
                k_split = k.split('_')
                if len(loss_terms.intersection(k_split)):
                    plt.ylim(0, None)
                elif len(metric_terms.intersection(k_split)):
                    #plt.ylim(0, 1)
                    plt.ylim(np.floor(ymin*10)/10, np.ceil(ymax*10)/10)
                    #plt.hlines([0.5,0.8,0.9], xmin, xmax, linestyles='-.', linewidth=1)
            plt.title(k)
            plt.legend()
            plt.show()
        else:
            #print(k+' no values')
            plt.close()


class AdamAccumulate(Optimizer):
    """Adam optimizer with accumulated gradients for having a virtual batch size larger 
    than the physical batch size.

    # Notes
        Default parameters follow those provided in the original paper.
        Only works with TensorFlow 1.x!

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        accum_iters: Number of batches between parameter update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, accum_iters=10, **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.accum_iters = K.variable(accum_iters)
    
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = ms + vs
        
        flag = K.equal(t % self.accum_iters, 0)
        flag = K.cast(flag, dtype='float32')
        
        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))
            
            # apply constraints.
            new_p = p_t
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates
            
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

