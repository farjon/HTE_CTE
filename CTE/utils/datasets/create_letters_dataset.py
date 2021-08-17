import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from GetEnvVar import GetEnvVar
from sklearn.preprocessing import QuantileTransformer
import torch

def main(args=None):

    # read 'LETTER' dataset and split to train-val-test
    Letter_data = pd.read_csv(os.path.join(args.datadir, 'LETTER.csv'), header=None)
    labels = Letter_data.loc[:,0]
    labels = (labels.apply(ord) - ord('A')).to_numpy()
    features = Letter_data.loc[:,1:].to_numpy().astype(np.float)
    x_train = features[:16000, :]
    x_test = features[16000:, :]
    y_train = labels[:16000]
    y_test = labels[16000:]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)

    X = {'train':x_train, 'val':x_val, 'test':x_test}
    # same procedure as in rtdl paper
    if args.normalization == 'quantile':
        normalizer = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
                                     copy= False)
        noise = 1e-3
        X_train = X['train']
        stds = np.std(X_train, axis=0, keepdims=True)
        noise_std = noise / np.maximum(stds, noise)
        X_train += noise_std * np.random.randn(*X_train.shape)
        normalizer.fit(X_train)
        {k: normalizer.transform(v) for k, v in X.items()}

    x_train = torch.from_numpy(X['train'])
    x_val = torch.from_numpy(X['val'])
    x_test = torch.from_numpy(X['test'])

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(x_train, os.path.join(args.output_path, 'x_train.pt'))
    torch.save(x_val, os.path.join(args.output_path, 'x_val.pt'))
    torch.save(x_test, os.path.join(args.output_path, 'x_test.pt'))
    torch.save(y_train, os.path.join(args.output_path, 'y_train.pt'))
    torch.save(y_val, os.path.join(args.output_path, 'y_val.pt'))
    torch.save(y_test, os.path.join(args.output_path, 'y_test.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTE model")
    args = parser.parse_args()
    args.normalization = 'quantile'
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'LETTER')
    args.output_path = os.path.join(GetEnvVar('DatasetsPath'), 'HTE Guy dataset', 'HTE_data', 'letter')
    main(args)