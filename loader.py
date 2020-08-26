import pickle
import os
import random
import numpy as np
import cv2
from albumentations import CoarseDropout, ShiftScaleRotate, HorizontalFlip


def load_batch(file_path, utf8=False):
    with open(file_path, 'rb') as file:
        if utf8:
            dataset = pickle.load(file, encoding='utf-8')
        else:
            dataset = pickle.load(file, encoding='bytes')
        return dataset


class DataManager:
    def __init__(self, dir, normalize=False):
        self.normalize = normalize
        self.dir = dir
        self.current_dataset = ''
        self.train_input = None
        self.train_label = None
        self.test_input = None
        self.test_label = None
        self.categories = None
        self.val_input = None
        self.val_label = None
        self.subset_input = None
        self.subset_label = None
        self._sample_idx = None
        self._pos = None
        self.mean = 0
        self.std = 1

    @property
    def num_class(self):
        assert self.current_dataset!='', 'Please Load Dataset First.'
        return len(self.categories)

    @property
    def dataset_name(self):
        return self.current_dataset

    def load_CIFAR_100(self, normalize=False, NHWC= False, fine_label=False):
        self.current_dataset = 'cifar-100'

        file_path = os.path.join(self.dir, self.current_dataset)
        dataset = load_batch(os.path.join(file_path, 'train'))
        self.train_input = np.array(dataset[b'data']).astype('float32')
        if fine_label:
            self.train_label = np.array(dataset[b'fine_labels']).astype('int32')
        else:
            self.train_label = np.array(dataset[b'coarse_labels']).astype('int32')
        dataset = load_batch(os.path.join(file_path, 'test'))
        self.test_input = np.array(dataset[b'data']).astype('float32')
        if fine_label:
            self.test_label = np.array(dataset[b'fine_labels']).astype('int32')
        else:
            self.test_label = np.array(dataset[b'coarse_labels']).astype('int32')
        dataset = load_batch(os.path.join(file_path, 'meta'), utf8=True)
        if fine_label:
            self.categories = dataset['fine_label_names']
        else:
            self.categories = dataset['coarse_label_names']

        self.train_input = self.train_input.reshape([50000, 3, 32, 32])
        self.test_input = self.test_input.reshape([10000, 3, 32, 32])
        if normalize:
            all_mask = np.concatenate([self.train_input, self.test_input])
            mean = all_mask.mean(axis=(0, 2, 3))[:, None, None]
            std = all_mask.std(axis=(0, 2, 3))[:, None, None]
            self.train_input = (self.train_input - mean) / std
            self.test_input = (self.test_input - mean) / std
        if NHWC:
            self.train_input = self.train_input.transpose([0, 2, 3, 1])
            self.test_input = self.test_input.transpose([0, 2, 3, 1])
        self.w = 32
        self.h = 32
        self.c = 3

    def load_MNIST(self, NHWC= False):
        self.current_dataset = 'mnist'
        self.categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        file_path = os.path.join(self.dir, self.current_dataset)
        with open(os.path.join(file_path, 'train-images.idx3-ubyte'), 'rb') as file:
            magic = int.from_bytes(file.read(4), byteorder='big', signed=False)
            num_img = int.from_bytes(file.read(4), byteorder='big', signed=False)
            h = int.from_bytes(file.read(4), byteorder='big', signed=False)
            w = int.from_bytes(file.read(4), byteorder='big', signed=False)
            self.train_input = np.array(bytearray(file.read()),dtype=np.float32).reshape([num_img, 1, h, w])
        with open(os.path.join(file_path, 'train-labels.idx1-ubyte'), 'rb') as file:
            magic = int.from_bytes(file.read(4), byteorder='big', signed=False)
            num_label = int.from_bytes(file.read(4), byteorder='big', signed=False)
            self.train_label = np.array(bytearray(file.read()),dtype=np.int32)
        with open(os.path.join(file_path, 't10k-images.idx3-ubyte'), 'rb') as file:
            magic = int.from_bytes(file.read(4), byteorder='big', signed=False)
            num_img = int.from_bytes(file.read(4), byteorder='big', signed=False)
            h = int.from_bytes(file.read(4), byteorder='big', signed=False)
            w = int.from_bytes(file.read(4), byteorder='big', signed=False)
            self.test_input = np.array(bytearray(file.read()),dtype=np.float32).reshape([num_img, 1, h, w])
        with open(os.path.join(file_path, 't10k-labels.idx1-ubyte'), 'rb') as file:
            magic = int.from_bytes(file.read(4), byteorder='big', signed=False)
            num_label = int.from_bytes(file.read(4), byteorder='big', signed=False)
            self.test_label = np.array(bytearray(file.read()),dtype=np.int32)
        self.mean = self.train_input.mean(axis=(0, 2, 3))[:, None, None]
        self.std = self.train_input.std(axis=(0, 2, 3))[:, None, None]
        if NHWC:
            self.train_input = self.train_input.transpose([0, 2, 3, 1])
            self.test_input = self.test_input.transpose([0, 2, 3, 1])
        self.w = w
        self.h = h
        self.c = 1

    def load_CIFAR_10(self, use_mean_and_std=None, NHWC= False):
        self.current_dataset = 'cifar-10-batches-py'
        file_path = os.path.join(self.dir, self.current_dataset)

        datasets = [load_batch(os.path.join(file_path, 'data_batch_{0}'.format(i+1))) for i in range(5)]
        self.train_label = np.concatenate([dataset[b'labels'] for dataset in datasets]).astype('long')
        self.train_input = np.concatenate([dataset[b'data'] for dataset in datasets]).astype('float32')
        # load test data
        dataset = load_batch(os.path.join(file_path, 'test_batch'))
        self.test_input = np.array(dataset[b'data']).astype('float32')
        self.test_label = np.array(dataset[b'labels']).astype('long')
        # load categories
        dataset = load_batch(os.path.join(file_path, 'batches.meta'), utf8=True)
        self.categories = dataset['label_names']

        self.train_input = self.train_input.reshape([50000, 3, 32, 32])
        self.test_input = self.test_input.reshape([10000, 3, 32, 32])
        if self.normalize:
            self.train_input = self.train_input / 255.0
            self.test_input = self.test_input / 255.0

        self.mean = self.train_input.mean(axis=(0, 2, 3))[:, None, None]
        self.std = self.train_input.std(axis=(0, 2, 3))[:, None, None]

        if use_mean_and_std is not None:
            self.mean, self.std = np.array(use_mean_and_std)
            self.mean = self.mean[:, None, None]
            self.std = self.std[:, None, None]

        self.train_input = (self.train_input - self.mean) / self.std
        self.test_input = (self.test_input - self.mean) / self.std
        if NHWC:
            self.train_input = self.train_input.transpose([0, 2, 3, 1])
            self.test_input = self.test_input.transpose([0, 2, 3, 1])
        self.w = 32
        self.h = 32
        self.c = 3

    def generate_val_and_subset(self):
        assert self.current_dataset!='', 'Please Load Dataset First.'
        num_class = len(self.categories)
        all_index = []
        masks = []
        for i in range(num_class):
            all_index.append(list(np.flatnonzero(self.train_label == i)) )
        for index in all_index:
            # if self.debug:
            #     masks.append(index)
            # else:
            masks.append(random.sample(index, len(index)))
        num_val = 5 if num_class == 100 else 500
        self.val_input = np.concatenate([self.train_input[mask[:num_val]] for mask in masks]).astype('float32')
        self.val_label = np.concatenate([self.train_label[mask[:num_val]] for mask in masks]).astype('long')
        if num_class != 100:
            self.subset_input = np.concatenate([self.train_input[mask[num_val:num_val + 500]] for mask in masks])
            self.subset_label = np.concatenate([self.train_label[mask[num_val:num_val + 500]] for mask in masks])
        self.train_input = np.concatenate([self.train_input[mask[num_val:]] for mask in masks]).astype('float32')
        self.train_label = np.concatenate([self.train_label[mask[num_val:]] for mask in masks]).astype('long')

    def augment(self, x):
        x = x.transpose(1, 2, 0)
        # cutout
        x = CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,
                          min_height=1, min_width=1, fill_value=0, p=0.5)(image=x)["image"]

        # width height shift and rotate
        x = ShiftScaleRotate(shift_limit=0.25, scale_limit=0, rotate_limit=0,
                             interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                             value=0, mask_value=0, p=0.5, )(image=x)["image"]

        # flip
        x = HorizontalFlip(p=0.5)(image=x)["image"]
        x = x.transpose(2, 0, 1)
        return (x - self.mean) / self.std

    def sample(self, batch_size, from_subset=False):
        assert self.current_dataset!='', 'Please Load Dataset First.'
        if from_subset:
            num_data = self.subset_input.shape[0]
            mask = np.random.choice(num_data, [batch_size])
            input_batch = self.subset_input[mask]
            label_batch = self.subset_label[mask]
            return input_batch, label_batch
        else:
            num_data = self.train_input.shape[0]
            if self._pos is None or self._sample_idx is None or self._pos >= num_data:
                self._sample_idx = np.random.choice(num_data, [num_data], replace=False)
                self._pos = 0
            if self._pos + batch_size >=num_data:
                mask = self._sample_idx[-batch_size:]
            else:
                mask = self._sample_idx[self._pos: self._pos + batch_size]
            input_batch = self.train_input[mask]
            # for i, x in enumerate(input_batch):
            #     input_batch[i] = self.augment(x)
            label_batch = self.train_label[mask]
            self._pos += batch_size
            return input_batch, label_batch

    # def to_tensor(self, type='torch'):
    #     self.train_input = torch.from_numpy(self.train_input)
    #     self.train_label = torch.from_numpy(self.train_label)
    #     self.val_input = torch.from_numpy(self.val_input)
    #     self.val_label = torch.from_numpy(self.val_label)
    #     self.test_input = torch.from_numpy(self.test_input)
    #     self.test_label = torch.from_numpy(self.test_label)
# if __name__=='__main__':
#     #import matplotlib.pyplot as plt
#     dm = DataManager('/home/liu/WORKSPACE/DATASET')
#     # dm.load_CIFAR_10()
#     # dm.load_MNIST()
#     dm.load_CIFAR_10(normalize=True)
#     dm.generate_val_and_subset()
#     # batch_x, batch_y = dm.sample(16)
#     # fig = plt.figure()
#     # for i in range(16):
#     #     ax = fig.add_subplot(4,4,i+1)
#     #     ax.imshow(batch_x[i].transpose([1,2,0]))
#     #     ax.set_title(dm.categories[batch_y[i]])
    # plt.show()
