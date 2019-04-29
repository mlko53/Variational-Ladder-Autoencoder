from dataset import *
import os
from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.datasets.cifar10 import load_data

class CIFAR10Dataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CIFAR10_data')
        cifar10 = load_data()
        self.train_image = np.clip(cifar10[0][0]/255., a_min=0.0, a_max=1.0)
        self.test_image = np.clip(cifar10[1][0]/255., a_min=0.0, a_max=1.0)
        self.name = "cifar10"
        self.data_dims = [32, 32, 3]
        self.train_size = 50000
        self.test_size = 10000
        self.range = [0.0, 1.0]
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0

    def next_batch(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.train_image.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
        return self.train_image[prev_batch_ptr:self.train_batch_ptr,:,:,:]

    def next_test_batch(self, batch_size):
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr > self.test_image.shape[0]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        return self.test_image[prev_batch_ptr:self.test_batch_ptr,:,:,:]

    def batch_by_index(self, batch_start, batch_end):
        return self.train_image[:, :, :, batch_start:batch_end]

    def display(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0


if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    images = dataset.next_batch(100)
    import pdb
    pdb.set_trace()
