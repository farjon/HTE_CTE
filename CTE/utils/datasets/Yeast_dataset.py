import torch
import os
import numpy as np
import pandas as pd

class Yeast(torch.utils.data.Dataset):
    def __init__(self, path_to_folder, mean = None, std = None):

        self.labels, self.examples = self.read_labels_and_ids(path_to_folder)
        if mean is None and std is None:
            self.mean = self.examples.mean().values
            self.std = self.examples.std().values
        else:
            self.mean = mean
            self.std = std

    def read_labels_and_ids(self, path_to_folder):
        labels = pd.read_csv(os.path.join(path_to_folder, 'labels.csv'))
        examples = pd.read_csv(os.path.join(path_to_folder, 'features.csv'))
        return labels, examples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        example = np.float32(self.examples.loc[index].values)
        example = np.float32((example - self.mean)/self.std)
        labels = self.labels.loc[index].values
        labels = torch.from_numpy(np.asarray(labels)).__long__()
        return example, labels