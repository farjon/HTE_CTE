import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def main(args=None):
    if os.path.isdir(args.datapath):
        train_path = os.path.join(args.datapath, 'train')
        test_path = os.path.join(args.datapath, 'test')
        val_path = os.path.join(args.datapath, 'val')
    else:
        # read 'LETTER' dataset and split to train-val-test
        Letter_data = pd.read_csv(os.path.join(args.datadir, 'LETTER.csv'), header=None)
        labels = Letter_data.loc[:,0]
        labels = (labels.apply(ord) - ord('A')).to_numpy()
        features = Letter_data.loc[:,1:].to_numpy().astype(np.float)
        train_features = features[:16000, :]
        test_features = features[16000:, :]
        train_labels = labels[:16000]
        test_labels = labels[16000:]
        x_train, x_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, shuffle=False)
        os.makedirs(args.datapath, exist_ok=True)
        train_path = os.path.join(args.datapath, 'train')
        test_path = os.path.join(args.datapath, 'test')
        val_path = os.path.join(args.datapath, 'val')
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        pd.DataFrame(x_train).to_csv(os.path.join(train_path, 'features.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(train_path, 'labels.csv'), index=False)
        pd.DataFrame(x_val).to_csv(os.path.join(val_path, 'features.csv'), index=False)
        pd.DataFrame(y_val).to_csv(os.path.join(val_path, 'labels.csv'), index=False)
        pd.DataFrame(test_features).to_csv(os.path.join(test_path, 'features.csv'), index=False)
        pd.DataFrame(test_labels).to_csv(os.path.join(test_path, 'labels.csv'), index=False)

    return train_path, val_path, test_path


if __name__ == '__main__':
    main()