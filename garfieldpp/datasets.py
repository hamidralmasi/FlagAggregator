
# Datasets management and partitioning.

#!/usr/bin/env python

import pathlib
import torch
import sys
from random import Random
from torchvision import datasets, transforms

# sharing_strategy = "file_descriptor" #file_system
# torch.multiprocessing.set_sharing_strategy(sharing_strategy)

# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)

datasets_list = ['mnist', 'cifar10', 'cifar10noisy', 'tinyimagenet', 'mnistnoisy','fmnist','tinyimagenetnoisy']
MNIST = datasets_list.index('mnist')
FMNIST = datasets_list.index('fmnist')
CIFAR10 = datasets_list.index('cifar10')
CIFAR10NOISY = datasets_list.index('cifar10noisy')
TINYIMAGENET = datasets_list.index('tinyimagenet')
TINYIMAGENETNOISY = datasets_list.index('tinyimagenetnoisy')
MNISTNOISY = datasets_list.index('mnistnoisy')

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       """ Constructor of Partition Object
           Args
           data		dataset needs to be partitioned
           index	indices of datapoints that are returned
        """
       self.data = data
       self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """ Fetching a datapoint given some index
	    Args
            index	index of the datapoint to be fetched
        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        """ Constructor of dataPartitioner object
	    Args
	    data	dataset to be partitioned
	    sizes	Array of fractions of each partition. Its contents should sum to 1
	    seed	seed of random generator for shuffling the data
	"""
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        """ Fetch some partition in the dataset
	    Args
	    partition	index of the partition to be fetched from the dataset
	"""
        return Partition(self.data, self.partitions[partition])

class DatasetManager(object):
    """ Manages training and test sets"""

    def __init__(self, dataset, augmentedfolder, minibatch, num_workers, size, rank):
        """ Constrctor of DatasetManager Object
	    Args
		dataset		dataset name to be used
		minibatch	minibatch size to be employed by each worker
		num_workers	number of works employed in the setup
		size		FIXME
		rank		rank of the current worker
	"""
        if dataset not in datasets_list:
            print("Existing datasets are: ", datasets_list)
            raise
        self.dataset = datasets_list.index(dataset)
        self.augmentedfolder = augmentedfolder
        self.batch = minibatch * num_workers
        self.num_workers = num_workers
        self.num_ps = size - num_workers
        self.rank = rank

    def fetch_dataset(self, dataset, train=True):
        """ Fetch train or test set of some dataset
		Args
		dataset		dataset index from the global "datasets" array
		train		boolean to determine whether to fetch train or test set
	"""
        homedir = str(pathlib.Path.home())


        if dataset == FMNIST:
            if train:
              transforms_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
              dataset = datasets.ImageFolder(homedir+'/data/FashionMNIST/train', transform=transforms_train)
              return dataset
            else:
              transforms_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),])
              dataset = datasets.ImageFolder(homedir+'/data/FashionMNIST/test', transform=transforms_test)
              return dataset

        if dataset == MNIST:
            if train:
              transforms_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
              dataset = datasets.ImageFolder(homedir+'/data/MNIST/train', transform=transforms_train)
              return dataset
            else:
              transforms_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),])
              dataset = datasets.ImageFolder(homedir+'/data/MNIST/test', transform=transforms_test)
              return dataset

        if dataset == MNISTNOISY:
            folder = "/data/MNISTnoisy/"+self.augmentedfolder
            if train:
              transforms_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_train)
              return dataset
            else:
              transforms_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),])
              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_test)
              return dataset

        if dataset == CIFAR10:
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
#                transforms.Resize(299),		#only use with inception
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              return datasets.CIFAR10(
               homedir+'/data',
               train=True,
               download=True,
               transform=transforms_train)
            else:
              transforms_test = transforms.Compose([
#                transforms.Resize((299,299)),			#only use with inception
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              return datasets.CIFAR10(
                homedir+'/data',
                train=False,
                download=True,
                transform=transforms_test)


        if dataset == CIFAR10NOISY:
            folder = "/data/cifar10noisy/"+self.augmentedfolder
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_train)
              return dataset

            else:
              transforms_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_test)
              return dataset

        if dataset == TINYIMAGENET:
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

              dataset = datasets.ImageFolder(homedir+'/data/tiny-imagenet-200/train', transform=transforms_train)

              return dataset
            else:
              transforms_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              dataset = datasets.ImageFolder(homedir+'/data/tiny-imagenet-200/test', transform=transforms_test)
              return dataset

        if dataset == TINYIMAGENETNOISY:
            folder = "/data/augmented_tiny/"+self.augmentedfolder
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_train)

              return dataset
            else:
              transforms_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              dataset = datasets.ImageFolder(homedir+folder, transform=transforms_test)
              return dataset


    def get_train_set(self):
        """ Fetch my partition of the train set"""
        train_set = self.fetch_dataset(self.dataset, train=True)
        size = self.num_workers
        bsz = int(self.batch / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_set, partition_sizes)
        partition = partition.use(self.rank - self.num_ps)
        print("Using batch size = ", bsz)
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=False, pin_memory=True)  # , worker_init_fn=set_worker_sharing_strategy
        return [sample for sample in train_set]

    def get_test_set(self):
        """ Fetch test set, which is global, i.e., same for all entities in the deployment"""
        test_set = self.fetch_dataset(self.dataset, train=False)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=100, #len(test_set),
		        pin_memory=True, shuffle=False, num_workers=2)  # , worker_init_fn=set_worker_sharing_strategy
        return test_set
