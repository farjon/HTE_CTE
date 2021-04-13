import numpy as np
import pandas as pd
import cv2
from GetEnvVar import GetEnvVar
import os
import matplotlib.pyplot as plt

def create_syn_data(n, means, covs, ratio = 0.5, train_test_split_factor = 0.8):
    '''
    this function creates synthetic data generated from Gaussian distribution
    the data will be generated from 2 different distributions for classification tasks
    :param n: the number of examples to generate in total
    :param D_in: the number of features for each example
    :param ratio: ration between the two classes. defualt is 0.5
    :return: x, y - examples and labels. size(x) = n*D_in and size(y) = n
    '''
    # dummy data - data will be generated from noise. a single feature will fully separate between the classes
    num_of_eaxmples_in_each_class = int(np.round(n*ratio))

    x_1 = np.random.multivariate_normal(means[0], covs[0], num_of_eaxmples_in_each_class)
    x_2 = np.random.multivariate_normal(means[1], covs[1], num_of_eaxmples_in_each_class)


    split_ind = int(num_of_eaxmples_in_each_class*train_test_split_factor)
    x1_train = x_1[:split_ind, :]
    x1_test = x_1[split_ind:, :]

    split_ind = int(num_of_eaxmples_in_each_class * train_test_split_factor)
    x2_train = x_2[:split_ind, :]
    x2_test = x_2[split_ind:, :]

    x_train = np.concatenate([x1_train,x2_train])
    x_test = np.concatenate([x1_test,x2_test])
    y_train = np.concatenate([np.zeros(x1_train[:,0].size), np.ones(x2_train[:,0].size)])
    y_test = np.concatenate([np.zeros(x1_test[:,0].size), np.ones(x2_test[:,0].size)])

    shuffle_train = np.random.permutation(x_train[:,0].size)
    shuffle_test = np.random.permutation(x_test[:,0].size)
    x_train = x_train[shuffle_train]
    x_test = x_test[shuffle_test]
    y_train = y_train[shuffle_train]
    y_test = y_test[shuffle_test]

    return x_train, x_test, y_train, y_test

def create_pariti_bit_syn_data(n, means, covs):
    x = np.random.multivariate_normal(means, covs, n)
    binary_x = np.zeros_like(x)
    binary_x[x < 0] = 0
    binary_x[x >= 0] = 1
    sum_binary_x = np.sum(binary_x, 1)
    y = sum_binary_x % 2
    return np.float32(x), y

def create_circular_mask(h, w):
    assert w == h, 'circle creation is only valid within a square'

    center = (int(w/2), int(h/2))
    radius = center[0]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = np.zeros([h,w]) - 1
    mask[dist_from_center <= radius] = 1
    return mask

def create_rectangle_and_circle_dataset_exp1(H = 50, W = 50, examples_for_each_class = 200, patch_size = 17):
    path_to_folder = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 1')
    square = np.ones([patch_size,patch_size])*255
    circle = create_circular_mask(patch_size,patch_size)*255

    for set in ['train', 'test']:
        labels = pd.DataFrame(columns=['file_name', 'class'])
        if set is 'train':
            num_of_images = int(examples_for_each_class*0.8)
        else:
            num_of_images = int(examples_for_each_class*0.2)
        path_to_save = os.path.join(path_to_folder, set, 'images')
        os.makedirs(path_to_save, exist_ok=True)
        for i in range(num_of_images):
            file_name = 's_' + str(i) + '.jpg'
            image = np.zeros([H,W])
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size] = square
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 0}, ignore_index=True)

        for i in range(num_of_images):
            file_name = 'c_' + str(i) + '.jpg'
            image = np.zeros([H,W])
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size] = circle
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 1}, ignore_index=True)

        labels.to_csv(os.path.join(path_to_folder, set, 'labels.csv'), index=False)

def create_rectangle_and_circle_dataset_exp2(H = 50, W = 50, examples_for_each_class = 400):
    path_to_folder = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 2')
    square_size = [9,13,17]
    square = []
    circle = []
    for patch_size in square_size:
        square.append(np.ones([patch_size,patch_size])*255)
        circle.append(create_circular_mask(patch_size,patch_size)*255)

    for set in ['train', 'test']:
        labels = pd.DataFrame(columns=['file_name', 'class'])
        if set is 'train':
            num_of_images = int(examples_for_each_class*0.8)
        else:
            num_of_images = int(examples_for_each_class*0.2)
        path_to_save = os.path.join(path_to_folder, set, 'images')
        os.makedirs(path_to_save, exist_ok=True)
        for i in range(num_of_images):
            file_name = 's_' + str(i) + '.jpg'
            image = np.zeros([H,W])
            patch_size_index = int(np.random.randint(0,len(square_size), 1))
            patch_size = square_size[patch_size_index]
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size] = square[patch_size_index]
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 0}, ignore_index=True)

        for i in range(num_of_images):
            file_name = 'c_' + str(i) + '.jpg'
            image = np.zeros([H,W])
            patch_size_index = int(np.random.randint(0,len(square_size), 1))
            patch_size = square_size[patch_size_index]
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size] = circle[patch_size_index]
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 1}, ignore_index=True)

        labels.to_csv(os.path.join(path_to_folder, set, 'labels.csv'), index=False)

def create_rectangle_and_circle_dataset_exp3(H = 100, W = 100, examples_for_each_class = 500, patch_size = 13):
    path_to_folder = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 3')
    square = np.ones([patch_size,patch_size])*255
    circle = create_circular_mask(patch_size,patch_size)*255

    for set in ['train', 'test']:
        labels = pd.DataFrame(columns=['file_name', 'class'])
        if set is 'train':
            num_of_images = int(examples_for_each_class*0.8)
        else:
            num_of_images = int(examples_for_each_class*0.2)
        path_to_save = os.path.join(path_to_folder, set, 'images')
        os.makedirs(path_to_save, exist_ok=True)
        class_dist = np.zeros(2)
        s_index = 0
        c_index = 0
        for i in range(num_of_images):
            small_class = np.random.randint(2)
            class_dist[small_class] += 1
            num_examples_for_small_class = np.random.randint(1, 4)
            image = np.zeros([H,W])
            shape_index = 0
            while shape_index != num_examples_for_small_class:
                x_min, y_min = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
                x_max, y_max = x_min + patch_size, y_min + patch_size
                if image[y_min, x_min] == 0 and image[y_max-1, x_min] == 0 and image[y_min, x_max-1] == 0 and image[y_max-1, x_max-1] == 0:
                    if small_class == 0:
                        image[y_min:y_max, x_min:x_max] = square
                    else:
                        image[y_min:y_max, x_min:x_max] = circle
                    shape_index += 1
            shape_index = 0
            while shape_index != (10 - num_examples_for_small_class):
                x_min, y_min = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
                x_max, y_max = x_min + patch_size, y_min + patch_size
                if image[y_min, x_min] == 0 and image[y_max-1, x_min] == 0 and image[y_min, x_max-1] == 0 and image[y_max-1, x_max-1] == 0:
                    if 1-small_class == 0:
                        image[y_min:y_max, x_min:x_max] = square
                    else:
                        image[y_min:y_max, x_min:x_max] = circle
                    shape_index += 1
            image[image<0] = 0
            if small_class == 1:
                file_name = 's_' + str(s_index) + '.jpg'
                s_index += 1
            else:
                file_name = 'c_' + str(c_index) + '.jpg'
                c_index += 1
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 1 - small_class}, ignore_index=True)

        labels.to_csv(os.path.join(path_to_folder, set, 'labels.csv'), index=False)


def create_rectangle_and_circle_dataset_exp4(H=50, W=50, examples_for_each_class=500, patch_size=17):
    path_to_folder = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 4')
    square_1 = np.ones([patch_size, patch_size, 1]) * 255

    square_RG = np.zeros([patch_size, patch_size, 1])
    square_GB = np.zeros([patch_size, patch_size, 1])
    square_RB = np.zeros([patch_size, patch_size, 1])

    # cv2 color is BGR
    square_RG = np.dstack((square_RG, square_1, square_1))
    square_GB = np.dstack((square_1, square_1, square_GB))
    square_RB = np.dstack((square_1, square_RB, square_1))
    square_RGB = np.ones([patch_size, patch_size, 3]) * 255

    for set in ['train', 'test']:
        labels = pd.DataFrame(columns=['file_name', 'class'])
        if set is 'train':
            num_of_images = int(examples_for_each_class*0.8)
        else:
            num_of_images = int(examples_for_each_class*0.2)
        path_to_save = os.path.join(path_to_folder, set, 'images')
        os.makedirs(path_to_save, exist_ok=True)
        for i in range(num_of_images):
            file_name = 'rg_s_' + str(i) + '.jpg'
            image = np.zeros([H,W,3])
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size, :] = square_RG
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 0}, ignore_index=True)

        for i in range(num_of_images):
            file_name = 'other_s_' + str(i) + '.jpg'
            image = np.zeros([H,W, 3])
            pixel_x, pixel_y = int(np.random.randint(0, H - patch_size + 1, 1)), int(np.random.randint(0, W - patch_size + 1, 1))
            if i%3 == 0:
                image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size, :] = square_RB
            elif i%3 == 1:
                image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size, :] = square_GB
            else:
                image[pixel_y:pixel_y+patch_size, pixel_x:pixel_x+patch_size, :] = square_RGB
            cv2.imwrite(os.path.join(path_to_save, file_name), image)
            labels = labels.append({'file_name': file_name, 'class': 1}, ignore_index=True)

        labels.to_csv(os.path.join(path_to_folder, set, 'labels.csv'), index=False)
if __name__ == '__main__':
    np.random.seed(10)
    create_rectangle_and_circle_dataset_exp1()