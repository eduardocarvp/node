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


class Dataset:

    def __init__(self, dataset, random_state, data_path='./data', normalize=False,
                 quantile_transform=False, output_distribution='normal', quantile_noise=0, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param random_state: global random seed for an experiment
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param quantile_transform: transforms the features to follow a normal distribution.
        :param output_distribution: if quantile_transform == True, data is projected onto this distribution
            See the same param of sklearn QuantileTransformer
        :param quantile_noise: if specified, fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable
            Please not that this transformation does NOT apply gaussian noise to the resulting data,
            the noise is only applied for QuantileTransformer
        :param kwargs: depending on the dataset, you may select train size, test size or other params
            If dataset is not in DATASETS, provide six keys: X_train, y_train, X_valid, y_valid, X_test and y_test
        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        if dataset in DATASETS:
            data_dict = DATASETS[dataset](os.path.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test')), \
                "Unknown dataset. Provide X_train, y_train, X_valid, y_valid, X_test and y_test params"
            data_dict = kwargs

        self.data_path = data_path
        self.dataset = dataset

        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_valid = data_dict['X_valid']
        self.y_valid = data_dict['y_valid']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']

        if all(query in data_dict.keys() for query in ('query_train', 'query_valid', 'query_test')):
            self.query_train = data_dict['query_train']
            self.query_valid = data_dict['query_valid']
            self.query_test = data_dict['query_test']

        if normalize:
            mean = np.mean(self.X_train, axis=0)
            std = np.std(self.X_train, axis=0)
            self.X_train = (self.X_train - mean) / std
            self.X_valid = (self.X_valid - mean) / std
            self.X_test = (self.X_test - mean) / std

        if quantile_transform:
            quantile_train = np.copy(self.X_train)
            if quantile_noise:
                stds = np.std(quantile_train, axis=0, keepdims=True)
                noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                quantile_train += noise_std * np.random.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=random_state, output_distribution=output_distribution).fit(quantile_train)
            self.X_train = qt.transform(self.X_train)
            self.X_valid = qt.transform(self.X_valid)
            self.X_test = qt.transform(self.X_test)

    def to_csv(self, path=None):
        if path == None:
            path = os.path.join(self.data_path, self.dataset)

        np.savetxt(os.path.join(path, 'X_train.csv'), self.X_train, delimiter=',')
        np.savetxt(os.path.join(path, 'X_valid.csv'), self.X_valid, delimiter=',')
        np.savetxt(os.path.join(path, 'X_test.csv'), self.X_test, delimiter=',')
        np.savetxt(os.path.join(path, 'y_train.csv'), self.y_train, delimiter=',')
        np.savetxt(os.path.join(path, 'y_valid.csv'), self.y_valid, delimiter=',')
        np.savetxt(os.path.join(path, 'y_test.csv'), self.y_test, delimiter=',')


def fetch_CLICK(path, valid_size=100_000, validation_seed=None):
    # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
    csv_path = os.path.join(path, 'click.csv')
    if not os.path.exists(csv_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

    data = pd.read_csv(csv_path, index_col=0)
    X, y = data.drop(columns=['target']), data['target']
    X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
    y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

    y_train = (y_train.values.reshape(-1) == 1).astype('int64')
    y_test = (y_test.values.reshape(-1) == 1).astype('int64')

    cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=validation_seed)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
    return dict(
        X_train=X_train.values.astype('float32'), y_train=y_train,
        X_valid=X_val.values.astype('float32'), y_valid=y_val,
        X_test=X_test.values.astype('float32'), y_test=y_test
    )


def preprocess(df, target, split_indices=None, 
              seed=42, normalize=True,
              quantile_transform=True,
              quantile_noise=0,
              output_distribution='normal'
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

    cat_features = X.columns[X.dtypes == object]

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_valid = (X_valid - mean) / std
        X_test = (X_test - mean) / std

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
        X_test=X_test.astype('float32'), y_test=y_test
    )