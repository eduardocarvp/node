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


class Model(BaseEstimator):
    def __init__(self, layer_dim=64, num_layers=1, tree_dim=4,
                 seed=42, device_name='auto'):
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim

        self.seed = seed
        torch.manual_seed(self.seed)
        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)

        print(f"Device used : {self.device}")

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            plot=False, early_stopping_rounds=10_000,
            report_frequency=100):

        num_features = X_train.shape[1]
        num_classes = len(set(y_train))

        self.network = Node(input_dim=num_features,
                            output_dim=num_classes,
                            layer_dim=self.layer_dim,
                            num_layers=self.num_layers,
                            tree_dim=self.tree_dim)
        self.network.to(self.device)

        trainer = Trainer(
            model=self.network, loss_function=F.cross_entropy,
            experiment_name='debug',
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=True,
            n_last_checkpoints=5
        )

        loss_history, auc_history = [], []
        best_val_auc = 0.
        best_step = 0

        for batch in iterate_minibatches(X_train, y_train, batch_size=1024, 
                                         shuffle=True, epochs=float('inf')):
            metrics = trainer.train_on_batch(*batch, device=self.device)
            
            loss_history.append(metrics['loss'])

            if trainer.step % report_frequency == 0:
                trainer.save_checkpoint()
                trainer.average_checkpoints(out_tag='avg')
                trainer.load_checkpoint(tag='avg')
                auc = trainer.evaluate_auc(
                    X_valid, y_valid, device=self.device, batch_size=1024)
                
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_step = trainer.step
                    trainer.save_checkpoint(tag='best')
                
                auc_history.append(auc)
                trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()
                
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
                
            if trainer.step > best_step + early_stopping_rounds:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step)
                print("Best Val AUC: %0.5f" % (best_val_auc))
                break

        trainer.load_checkpoint(tag='best')

    def predict(self, X):
        X = torch.as_tensor(X, device=self.device)
        self.network.train(False)

        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.network, X, batch_size=1024))
            logits = check_numpy(logits)

        return logits
