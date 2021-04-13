import torch
import csv
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class CS_Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_folder):

        self.labels, self.files_names = self.read_labels_and_ids(path_to_folder)
        self.path_to_images = os.path.join(path_to_folder, 'images')

    def read_labels_and_ids(self, path_to_folder):
        labels = {}
        files_names = []
        labels_file = os.path.join(path_to_folder, 'labels.csv')
        with open(labels_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                files_names.append(row['file_name'])
                labels[row['file_name']] = row['class']
        return labels, files_names

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, index):
        file_name = self.files_names[index]
        image = Image.open((os.path.join(self.path_to_images, file_name)))
        image = TF.to_tensor(image)
        labels = int(self.labels[file_name])
        labels = torch.from_numpy(np.asarray(labels)).__long__()
        return image, labels