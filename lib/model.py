import torch
import numpy as np
from abc import abstractmethod
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.nn.utils import clip_grad_norm_
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
import copy
from .arch import Node

import os, sys
import time
sys.path.insert(0, '..')
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam
from tqdm import tqdm

from .utils import (
    iterate_minibatches,
    process_in_chunks,
    check_numpy
)
from .trainer import Trainer
from IPython.display import clear_output
import matplotlib.pyplot as plt


class NodeClassifier(BaseEstimator):
    def __init__(self, input_dim, output_dim,
                 layer_dim=512, num_layers=1, tree_dim=4,
                 seed=42, device_name='auto',
                 cat_idxs=None, cat_dims=None, cat_emb_dim=2,
                 experiment_name='debug',
                 clean_logs=True):
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.experiment_name = experiment_name

        self.seed = seed
        torch.manual_seed(self.seed)
        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)

        num_features = input_dim
        num_classes = output_dim

        self.network = Node(input_dim=num_features,
                            output_dim=num_classes,
                            layer_dim=self.layer_dim,
                            num_layers=self.num_layers,
                            tree_dim=self.tree_dim,
                            cat_idxs=self.cat_idxs,
                            cat_dims=self.cat_dims,
                            cat_emb_dim=self.cat_emb_dim)
        self.network.to(self.device)

        self.trainer = Trainer(
            model=self.network, loss_function=F.cross_entropy,
            experiment_name=self.experiment_name,
            warm_start=False,
            Optimizer=torch.optim.Adam,  # QHAdam,
            optimizer_params=dict(lr=5e-3),  # dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=True,
            n_last_checkpoints=5
        )

        if clean_logs:
            self.trainer.remove_all_checkpoints()

        print(f"Device used : {self.device}")

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            plot=False, early_stopping_rounds=10_000,
            report_frequency=20):

        loss_history, auc_history = [], []
        best_val_auc = 0.
        best_step = 0

        for batch in iterate_minibatches(X_train, y_train, batch_size=1024, 
                                         shuffle=True, epochs=float('inf')):
            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            
            loss_history.append(metrics['loss'])

            if self.trainer.step % report_frequency == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')
                auc = self.trainer.evaluate_auc(
                    X_valid, y_valid, device=self.device, batch_size=1024)
                
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_step = self.trainer.step
                    self.trainer.save_checkpoint(tag='best')
                
                auc_history.append(auc)
                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()
                
                if plot:
                    clear_output(True)
                    plt.figure(figsize=[12, 6])
                    plt.subplot(1, 2, 1)
                    plt.plot(loss_history)
                    plt.grid()
                    plt.subplot(1,2,2)
                    plt.plot(auc_history)
                    plt.grid()
                    plt.show()

                    print("Loss %.5f" % (metrics['loss']))
                    print("Val AUC: %0.5f" % (auc))
                
            if self.trainer.step > best_step + early_stopping_rounds:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step)
                print("Best Val AUC: %0.5f" % (best_val_auc))
                break

        self.trainer.load_checkpoint(tag='best')

    def predict(self, X):
        X = torch.as_tensor(X, device=self.device)
        self.network.train(False)

        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.network, X, batch_size=1024), dim=1)
            logits = check_numpy(logits)

        return logits


class NodeRegressor(BaseEstimator):
    def __init__(self, input_dim, output_dim,
                 layer_dim=512, num_layers=1, tree_dim=6,
                 seed=42, device_name='auto',
                 cat_idxs=None, cat_dims=None, cat_emb_dim=2,
                 experiment_name='debug',
                 clean_logs=True):
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.experiment_name = experiment_name

        self.seed = seed
        torch.manual_seed(self.seed)
        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)

        num_features = input_dim

        self.network = Node(input_dim=input_dim,
                            output_dim=output_dim,
                            layer_dim=self.layer_dim,
                            num_layers=self.num_layers,
                            tree_dim=self.tree_dim,
                            cat_idxs=self.cat_idxs,
                            cat_dims=self.cat_dims,
                            cat_emb_dim=self.cat_emb_dim)
        self.network.to(self.device)

        self.trainer = Trainer(
            model=self.network, loss_function=F.mse_loss,
            experiment_name=self.experiment_name,
            warm_start=False,
            Optimizer=torch.optim.Adam,  # QHAdam,
            optimizer_params=dict(lr=5e-3),  # dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            # Optimizer=QHAdam,
            # optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=True,
            n_last_checkpoints=5
        )

        if clean_logs:
            self.trainer.remove_all_checkpoints()

        print(f"Device used : {self.device}")

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            plot=False, early_stopping_rounds=10_000,
            report_frequency=50):

        loss_history, mse_history = [], []
        best_val_mse = float('inf')
        best_step = 0

        with torch.no_grad():
            res = self.network(torch.as_tensor(X_train[:1000], device=self.device))
            # trigger data-aware init

        for batch in iterate_minibatches(X_train, y_train.astype(float), batch_size=1024, 
                                         shuffle=True, epochs=float('inf')):
            metrics = self.trainer.train_on_batch(*batch, device=self.device, experiment_type='regression')
            
            loss_history.append(metrics['loss'])

            if self.trainer.step % report_frequency == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')
                mse = self.trainer.evaluate_mse(
                    X_valid, y_valid, device=self.device, batch_size=1024)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_step = self.trainer.step
                    self.trainer.save_checkpoint(tag='best')
                
                mse_history.append(mse)
                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()
                
                if plot:
                    clear_output(True)
                    plt.figure(figsize=[12, 6])
                    plt.subplot(1, 2, 1)
                    plt.plot(loss_history)
                    plt.grid()
                    plt.subplot(1,2,2)
                    plt.plot(mse_history)
                    plt.grid()
                    plt.show()

                    print("Loss %.5f" % (metrics['loss']))
                    print("Val mse: %0.5f" % (mse))
                
            if self.trainer.step > best_step + early_stopping_rounds:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step)
                print("Best Val mse: %0.5f" % (best_val_mse))
                break

        self.trainer.load_checkpoint(tag='best')

    def predict(self, X):
        X = torch.as_tensor(X, device=self.device)
        self.network.train(False)

        with torch.no_grad():
            logits = process_in_chunks(self.network, X, batch_size=1024)
            logits = check_numpy(logits)

        return logits
