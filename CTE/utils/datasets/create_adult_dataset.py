import os
from GetEnvVar import GetEnvVar
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


def main(args=None):
    if os.path.isdir(args.datapath):
        train_path = os.path.join(args.datapath, 'train')
        test_path = os.path.join(args.datapath, 'test')
        # val_path = os.path.join(args.datapath, 'val')
    else:
        # read 'ADULT' dataset and split to train-val-test
        adult_data = pd.read_csv(os.path.join(args.datadir, 'AdultLabelEncoding.csv'))
        # dropping non category columns
        adult_data = adult_data.drop([
            'workclass',
            'education',
            'marital.status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native.country',
            'income'
        ], axis='columns')

        classes = adult_data['income.cat']
        features = adult_data.drop(['income.cat'], axis='columns')
        x_train, x_test, y_train, y_test = train_test_split(features.to_numpy(), classes, test_size=0.2, shuffle=True)
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        os.makedirs(args.datapath, exist_ok=True)
        train_path = os.path.join(args.datapath, 'train')
        test_path = os.path.join(args.datapath, 'test')
        # val_path = os.path.join(args.datapath, 'val')
        os.makedirs(train_path, exist_ok=True)
        # os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        pd.DataFrame(x_train).to_csv(os.path.join(train_path, 'features.csv'), index=False, header=None)
        pd.DataFrame(y_train).to_csv(os.path.join(train_path, 'labels.csv'), index=False, header=None)
        # pd.DataFrame(x_val).to_csv(os.path.join(val_path, 'features.csv'), index=False, header=None)
        # pd.DataFrame(y_val).to_csv(os.path.join(val_path, 'labels.csv'), index=False, header=None)
        pd.DataFrame(x_test).to_csv(os.path.join(test_path, 'features.csv'), index=False, header=None)
        pd.DataFrame(y_test).to_csv(os.path.join(test_path, 'labels.csv'), index=False, header=None)

    return train_path, test_path
    # return train_path, val_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTE model")
    args = parser.parse_args()

    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'ADULT')
    args.datapath = os.path.join(args.datadir, 'split_data')
    args.train_rate = 0.8
    # args.val_rate = 0.2
    main(args)
