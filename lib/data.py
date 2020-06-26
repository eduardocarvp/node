import os
import bz2
import numpy as np
import pandas as pd
import gzip
import shutil
import torch
import random
import warnings

from sklearn.model_selection import train_test_split

from .utils import download
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import QuantileTransformer
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder


def preprocess(df, target, split_indices=None, 
              seed=42, normalize=True,
              quantile_transform=True,
              quantile_noise=0,
              output_distribution='normal',
              encoder='cat_encoder'
):
    X, y = df.drop(columns=[target]), df[target]
    
    # encode target
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(y.values)

    # split data
    if split_indices is None:
        X_train, X_valid, y_train, y_valid = train_test_split(
           X, y, test_size=0.2, random_state=seed
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid, y_valid, test_size=0.5, random_state=seed
        )
    else:
        X_train, y_train = X.loc[split_indices['train']], y[split_indices['train']]
        X_valid, y_valid = X.loc[split_indices['valid']], y[split_indices['valid']]
        X_test, y_test = X.loc[split_indices['test']], y[split_indices['test']]

    cat_features = X.columns[X.dtypes == object]
    cat_idxs = [i for i in range(len(X.columns)) if X.columns[i] in cat_features]
    cat_dims = []

    num_features = [feat for feat in X.columns if feat not in cat_features]

    print(num_features)

    if encoder == 'cat_encoder':
        cat_encoder = LeaveOneOutEncoder()
        cat_encoder.fit(X_train[cat_features], y_train)
        X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
        X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
        X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
    elif encoder == 'label_encoder':
        for col in cat_features:
            lab_encoder = LabelEncoder()
            lab_encoder.fit(X[col].fillna("VV_likely"))
            X_train[col] = lab_encoder.transform(X_train[col].fillna("VV_likely").values)
            X_valid[col] = lab_encoder.transform(X_valid[col].fillna("VV_likely").values)
            X_test[col] = lab_encoder.transform(X_test[col].fillna("VV_likely").values)
            cat_dims.append(len(lab_encoder.classes_))

    if normalize:
        mean = np.mean(X_train[num_features], axis=0)
        std = np.std(X_train[num_features], axis=0)
        X_train[num_features] = (X_train[num_features] - mean) / std
        X_valid[num_features] = (X_valid[num_features] - mean) / std
        X_test[num_features] = (X_test[num_features] - mean) / std

    if quantile_transform:
        quantile_train = np.copy(X_train)
        if quantile_noise:
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = quantile_noise / np.maximum(stds, quantile_noise)
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)

        qt = QuantileTransformer(random_state=seed, output_distribution=output_distribution).fit(quantile_train)
        X_train = qt.transform(X_train)
        X_valid = qt.transform(X_valid)
        X_test = qt.transform(X_test)

    return dict(
        X_train=X_train.astype('float32'), y_train=y_train,
        X_valid=X_valid.astype('float32'), y_valid=y_valid,
        X_test=X_test.astype('float32'), y_test=y_test,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
    )