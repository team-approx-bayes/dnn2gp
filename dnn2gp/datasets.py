import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

DEFAULT_DATA_FOLDER = 'data'


class Dataset:
    def __init__(self, data_set, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        mnist_train = transforms.Compose([
            transforms.ToTensor(),
        ])  # meanstd transformation

        mnist_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        if data_set == 'mnist':
            self.train_set = dset.MNIST(root=data_folder + '/' + data_set,
                                        train=True,
                                        transform=mnist_train,
                                        download=True)

            self.test_set = dset.MNIST(root=data_folder + '/' + data_set,
                                       train=False,
                                       transform=mnist_test)

        if data_set == 'cifar10':
            self.train_set = dset.CIFAR10(root=data_folder + '/' + data_set,
                                          train=True,
                                          transform=transform_test,
                                          download=True)

            self.test_set = dset.CIFAR10(root=data_folder + '/' + data_set,
                                         train=False,
                                         transform=transform_test)

    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=8)
        return train_loader

    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=8)
        return test_loader

    def load_full_train_set(self, use_cuda=torch.cuda.is_available()):

        full_train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=len(self.train_set),
                                       shuffle=False)

        x_train, y_train = next(iter(full_train_loader))

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_full_test_set(self, use_cuda=torch.cuda.is_available()):

        full_test_loader = DataLoader(dataset=self.test_set,
                                      batch_size=len(self.test_set),
                                      shuffle=False)

        x_test, y_test = next(iter(full_test_loader))

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test
