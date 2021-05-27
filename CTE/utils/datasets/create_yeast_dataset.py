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
        # read 'YEAST' dataset and split to train-val-test
        yeast_data = pd.read_csv(os.path.join(args.datadir, 'yeast_csv.csv'))
        classes = yeast_data['class_protein_localization'].unique()
        features = yeast_data.drop(['class_protein_localization'], axis='columns')
        train_features = pd.DataFrame(columns=features.keys())
        train_labels = pd.DataFrame()
        # val_features = pd.DataFrame(columns=features.keys())
        # val_labels = pd.DataFrame()
        test_features = pd.DataFrame(columns=features.keys())
        test_labels = pd.DataFrame()
        for i in range(len(classes)):
            class_name = classes[i]
            current_data = features[yeast_data['class_protein_localization'] == class_name]
            number_of_examples = len(current_data)
            # if number_of_examples  < 10:
            #     permuted_indices = np.random.permutation(number_of_examples)
            #     train_indices = permuted_indices[0:int(number_of_examples*0.6)]
            #     test_indices = permuted_indices[int(number_of_examples*0.6):]
            # else:
            #     permuted_indices = np.random.permutation(number_of_examples)
            #     train_indices = permuted_indices[0:int(number_of_examples*args.train_rate)]
            #     val_indices = permuted_indices[int(number_of_examples*args.train_rate):int(number_of_examples*args.train_rate) + int(number_of_examples*args.val_rate)]
            #     test_indices = permuted_indices[int(number_of_examples*args.train_rate) + int(number_of_examples*args.val_rate):]
            #
            #     val_features = val_features.append(current_data.iloc[val_indices], ignore_index=True)
            #     val_labels = val_labels.append(list(np.repeat(i, len(val_indices))), ignore_index=True)
            permuted_indices = np.random.permutation(number_of_examples)
            train_indices = permuted_indices[0:int(number_of_examples * 0.7)]
            test_indices = permuted_indices[int(number_of_examples * 0.7):]
            train_features = train_features.append(current_data.iloc[train_indices], ignore_index=True)
            test_features = test_features.append(current_data.iloc[test_indices], ignore_index=True)
            train_labels = train_labels.append(list(np.repeat(i, len(train_indices))), ignore_index=True)
            test_labels = test_labels.append(list(np.repeat(i, len(test_indices))), ignore_index=True)

        os.makedirs(args.datapath, exist_ok=True)
        train_path = os.path.join(args.datapath, 'train')
        test_path = os.path.join(args.datapath, 'test')
        # val_path = os.path.join(args.datapath, 'val')
        os.makedirs(train_path, exist_ok=True)
        # os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        pd.DataFrame(train_features).to_csv(os.path.join(train_path, 'features.csv'), index=False)
        pd.DataFrame(train_labels).to_csv(os.path.join(train_path, 'labels.csv'), index=False)
        # pd.DataFrame(val_features).to_csv(os.path.join(val_path, 'features.csv'), index=False)
        # pd.DataFrame(val_labels).to_csv(os.path.join(val_path, 'labels.csv'), index=False)
        pd.DataFrame(test_features).to_csv(os.path.join(test_path, 'features.csv'), index=False)
        pd.DataFrame(test_labels).to_csv(os.path.join(test_path, 'labels.csv'), index=False)

    return train_path, test_path
    # return train_path, val_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTE model")
    args = parser.parse_args()

    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'YEAST', 'yeast_zip', 'data')
    args.datapath = os.path.join(args.datadir, 'split_data')
    args.train_rate = 0.64
    # args.val_rate = 0.16
    main(args)