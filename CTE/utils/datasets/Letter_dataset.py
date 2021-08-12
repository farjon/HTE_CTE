import torch
import os
import numpy as np
import pandas as pd

class Letters(torch.utils.data.Dataset):
    def __init__(self, path_to_folder, set ='train', device = 'cpu'):

        self.labels, self.examples = self.read_labels_and_ids(path_to_folder, set, device)

    def read_labels_and_ids(self, path_to_folder, set, device):
        labels = torch.load(os.path.join(path_to_folder, f'y_{set}.pt'), map_location = device)
        examples = torch.load(os.path.join(path_to_folder,f'x_{set}.pt'), map_location = device)
        return labels, examples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        batch_examples = self.examples[index].to(torch.float32)
        batch_labels = self.labels[index]
        return batch_examples, batch_labels