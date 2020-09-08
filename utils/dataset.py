import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import warnings
import os
import os.path

from functools import reduce
from operator import __or__

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler



def onehot(k):
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def create_mnist_mnist(location="../data", n_labels=10, use_cnn=False):

    binary_fc = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()
    binary_cnn = lambda x: transforms.ToTensor()(x).bernoulli()

    greyscale_fc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    greyscale_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if use_cnn:
        # transformation = greyscale_cnn
        transformation = binary_cnn
    else:
        transformation = binary_fc

    mnist_train = MNIST(location, train=True, download=True,
                        transform=transformation, target_transform=onehot(n_labels), n_labels=n_labels)
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=transformation, target_transform=onehot(n_labels), n_labels=n_labels)

    return mnist_train, mnist_valid


def create_halfmoon_dataset(n_samples=1000, noise=0.1, random_state=0, to_plot=False):

    datasets = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X, y = datasets
    X = StandardScaler().fit_transform(X)
    X = X.astype(float)

    if to_plot:
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig, ax = plt.subplots()
        ax.set_title("Input data")

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                    edgecolors='k')

        ax.set_xticks(())
        ax.set_yticks(())

        plt.tight_layout()
        plt.show()

    return datasets

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        # print((self.indices[i] for i in torch.from_numpy(np.arange(len(self.indices)))))
        # return (self.indices[i] for i in torch.randperm(len(self.indices)))
        return (self.indices[i] for i in torch.from_numpy(np.arange(len(self.indices))))

    def __len__(self):
        return len(self.indices)


class HalfMoon(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, points, labels):
        'Initialization'
        self.labels = labels
        self.points = points

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.points[index]
        y = self.labels[index]

        return X, y, index


class MNIST(VisionDataset):
    """ Based on pytorch MNIST implementation but with __getitem__ return additional index.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, n_labels=10):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if n_labels == 2:
            # MNIST Binary Classification between digit a and b
            a = 7
            b = 9
            idxs = torch.where((self.targets == a) | (self.targets == b))
            self.data = self.data[idxs]
            self.targets = self.targets[idxs]

            a_idxs = torch.where(self.targets == a)
            b_idxs = torch.where(self.targets == b)

            self.targets[a_idxs] = 0
            self.targets[b_idxs] = 1


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class ActiveMoon:
    def __init__(self, datasets, use_cuda, valid_ratio=.4, batch_size=64, labels_per_class=10, seed=None):
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.labels_per_class = labels_per_class

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X, y = datasets
        X = StandardScaler().fit_transform(X)
        X = X.astype(float)
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X, y, test_size=valid_ratio, random_state=42)

        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

        self.train_len = len(y_train)

        # Randomly pick 10 points for each class as the first label training data
        indices = np.arange(self.train_len)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        # --------------
        self.labelled_indices = np.hstack(
            [list(filter(lambda idx: y_train[idx] == i, indices))[:labels_per_class] for i in range(2)])
        self.unlabelled_indices = np.setdiff1d(np.arange(self.train_len),
                                               self.labelled_indices)

        self.moon_train = HalfMoon(X_train, y_train)
        self.moon_valid = HalfMoon(X_valid, y_valid)

    def get_valid(self):
        'Get data for validation'
        return torch.utils.data.DataLoader(self.moon_valid, batch_size=self.batch_size, num_workers=1,
                                           pin_memory=self.use_cuda)

    def get_next_train(self, acquired_idx=None):
        'Get training data for next active learning loop'

        # If there are acquired data from last loop, update them (index of the data only)
        if acquired_idx is not None:
            self.labelled_indices = np.concatenate((self.labelled_indices, acquired_idx), axis=None)
            self.unlabelled_indices = np.setdiff1d(np.arange(self.train_len), self.labelled_indices)

        labelled_indices = torch.from_numpy(self.labelled_indices)
        unlabelled_indices = torch.from_numpy(self.unlabelled_indices)

        labelled = torch.utils.data.DataLoader(self.moon_train, batch_size=self.batch_size,
                                               num_workers=1, pin_memory=self.use_cuda,
                                               sampler=SubsetRandomSampler(labelled_indices))
        unlabelled = torch.utils.data.DataLoader(self.moon_train, batch_size=self.batch_size,
                                                 num_workers=1, pin_memory=self.use_cuda,
                                                 sampler=SubsetRandomSampler(unlabelled_indices))
        return labelled, unlabelled

    def get_xyuv_for_printing(self, labelled, unlabelled):
        'Get data for plotting the moons'
        x_lab = np.array([])
        y_lab = np.array([])
        u_unlab = np.array([])
        v_unlab = np.array([])
        for (x, y, _) in labelled:
            x_lab = np.vstack([x_lab, x.numpy()]) if x_lab.size else x.numpy()
            y_lab = np.concatenate((y_lab, y.numpy()), axis=None) if y_lab.size else y.numpy()
        for (u, v, _) in unlabelled:
            u_unlab = np.vstack([u_unlab, u.numpy()]) if u_unlab.size else u.numpy()
            v_unlab = np.concatenate((v_unlab, v.numpy()), axis=None) if v_unlab.size else v.numpy()

        return self.X_train, self.X_valid, self.y_train, self.y_valid, x_lab, y_lab, u_unlab, v_unlab


class ActiveMNIST:
    def __init__(self, mnist_train, mnist_valid, use_cuda, batch_size=64, labels_per_class=10,
                 n_labels=10, seed=None):
        """
        :param n_labels: The first n classes for classification
        """
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.labels_per_class = labels_per_class
        self.n_labels = n_labels

        self.train = mnist_train
        self.valid = mnist_valid

        self.train_len = len(mnist_train)

        # Randomly pick n points for the first n_labels class as the first label training data

        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [self.train.targets == i for i in np.arange(self.n_labels)]))
        self.train_indices = np.copy(indices)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        np.random.shuffle(indices)
        # --------------
        self.labelled_indices = np.hstack(
            [list(filter(lambda idx: self.train.targets[idx] == i, indices))[:labels_per_class]
                for i in np.arange(self.n_labels)])
        self.unlabelled_indices = np.setdiff1d(self.train_indices, self.labelled_indices)

    def get_valid(self):
        'Get data for validation'
        return torch.utils.data.DataLoader(self.valid, batch_size=self.batch_size, pin_memory=False)

    def get_next_train(self, acquired_idx=None):
        'Get training data for next active learning loop'

        # If there are acquired data from last loop, update them (index of the data only)
        if acquired_idx is not None:
            self.labelled_indices = np.concatenate((self.labelled_indices, acquired_idx), axis=None)
            self.unlabelled_indices = np.setdiff1d(self.train_indices, self.labelled_indices)

        labelled_indices = torch.from_numpy(self.labelled_indices)
        unlabelled_indices = torch.from_numpy(self.unlabelled_indices)

        labelled = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=False,
                                               sampler=SubsetRandomSampler(labelled_indices))
        unlabelled = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=False,
                                                 sampler=SubsetRandomSampler(unlabelled_indices))
        return labelled, unlabelled

    # def get_xyuv_for_printing(self, labelled, unlabelled):
    #     'Get data for plotting the moons'
    #     x_lab = np.array([])
    #     y_lab = np.array([])
    #     u_unlab = np.array([])
    #     v_unlab = np.array([])
    #     for (x, y, _) in labelled:
    #         x_lab = np.vstack([x_lab, x.numpy()]) if x_lab.size else x.numpy()
    #         y_lab = np.concatenate((y_lab, y.numpy()), axis=None) if y_lab.size else y.numpy()
    #     for (u, v, _) in unlabelled:
    #         u_unlab = np.vstack([u_unlab, u.numpy()]) if u_unlab.size else u.numpy()
    #         v_unlab = np.concatenate((v_unlab, v.numpy()), axis=None) if v_unlab.size else v.numpy()
    #
    #     return self.X_train, self.X_valid, self.y_train, self.y_valid, x_lab, y_lab, u_unlab, v_unlab