import os
import sys
import random
import numpy as np
from distutils.log import error
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFilter
from utils.randaugments import RandAugmentMC
# from make_dataset import CustomDataset

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target

class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, class_num=10, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()

        if hasattr(self.data, 'split'):
            if self.data.split == 'unlabeled':
                data_indexes = np.arange(len(self.data))
                random.Random().shuffle(data_indexes)
                sublen = len(data_indexes) // len(partition_sizes[0])
                last_suben = len(data_indexes) - (sublen * (len(partition_sizes[0]) - 1))
                self.partitions = [list()] + [data_indexes[i*sublen: (i+1)*sublen+last_suben] 
                                              for i in range(len(partition_sizes[0]))]

                return
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        try:
            for class_idx in range(len(data.classes)):
                label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        except AttributeError:
            for class_idx in range(class_num):
                label_indexes.append(list(np.where(np.array(data.labels) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        
        # distribute class indexes to each vm according to sizes matrix
        try:
            for class_idx in range(len(data.classes)):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx
        except AttributeError:
            for class_idx in range(class_num):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)

def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4, drop_last=False, persistent_workers=True):
    if selected_idxs == None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers,
                                    drop_last=drop_last, persistent_workers=persistent_workers)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers,
                                    drop_last=drop_last, persistent_workers=persistent_workers)
    
    return DataLoaderHelper(dataloader)

def load_datasets(dataset_type, data_path="zpsun/data", transform_type='default', **kwargs):
    if os.path.exists('/data0'):
        data_path = os.path.join('/data0', data_path)
    elif os.path.exists('/data'):
        data_path = os.path.join('/data', data_path)
    else:
        raise FileNotFoundError('Data Directory doesn not exist!')
    
    if transform_type == 'default':
        train_transform = load_default_transform(dataset_type, train=True)
    elif transform_type == 'twice':
        train_transform = load_default_transform(dataset_type, train=True)
        train_transform = TransformTwice(train_transform)
    elif transform_type == 'fixmatch':
        train_transform = RandTransform(dataset_type)
    elif transform_type == 'fixmatch_twice':
        train_transform = RandTransform(dataset_type, mode='fixmatch_strong2')
    elif transform_type == 'strong_twice':
        train_transform = RandTransform(dataset_type, mode='strong2')
    elif transform_type == 'strong':
        train_transform = RandTransform(dataset_type, mode='strong')                                
    else:
        raise error('Invalid transforamation type')
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'SVHN':
        train_dataset = datasets.SVHN(data_path+'/SVHN_data', split='train',
                                            download = True, transform=train_transform)
        test_dataset = datasets.SVHN(data_path+'/SVHN_data', split='test', 
                                            download = True, transform=test_transform)

    elif dataset_type == 'IMAGE100':
        train_dataset = datasets.ImageFolder(data_path+'/IMAGE100/train', transform=train_transform)
        test_dataset = datasets.ImageFolder(data_path+'/IMAGE100/test', transform=test_transform)

    elif dataset_type == "STL10":
        unlabeled_dataset = datasets.STL10(root=data_path, split='unlabeled', download=True,
                                                transform=train_transform)
        
        train_dataset = datasets.STL10(root=data_path, split='train', download=True,
                                                transform=train_transform)
        test_dataset = datasets.STL10(root=data_path, split='test', download=True,
                                    transform=test_transform)
        
        if 'unlabel' in kwargs.keys() and kwargs['unlabel']:
            return unlabeled_dataset, test_dataset
        
        return train_dataset, test_dataset, unlabeled_dataset

    return train_dataset, test_dataset


def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4, padding_mode='reflect'),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                    std=[0.5, 0.5, 0.5])
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=32,
                                                    padding=int(32*0.125),
                                                    padding_mode='reflect'),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'IMAGE100':
        if train:
            dataset_transform = transforms.Compose([
                                        transforms.Resize((144,144)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
        else:
            dataset_transform = transforms.Compose([
                                        transforms.Resize((144,144)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])

    elif dataset_type == "STL10":
        normalize = transforms.Normalize(mean=[0.4408, 0.4279, 0.3867],
                                        std=[0.2683, 0.2610, 0.2687])

        if train:
            dataset_transform = transforms.Compose([
                                    transforms.RandomCrop(96, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
        else:
            dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
    return dataset_transform


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class RandTransform(object):
    def __init__(self, dataset_type, mode='fixmatch'):
        resize = None
        if dataset_type == 'CIFAR10':
            hw = (32, 32)
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])
            
        elif dataset_type == 'SVHN':
            hw = (32, 32)
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                            std=[0.5, 0.5, 0.5])
        elif dataset_type == "STL10":
            hw = (96, 96)
            normalize = transforms.Normalize(mean=[0.4408, 0.4279, 0.3867],
                                            std=[0.2683, 0.2610, 0.2687])

        elif dataset_type == 'IMAGE100':
            hw = (144, 144)
            resize = transforms.Resize((144, 144))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
        else:
            raise error('RandTransform: Invalid transforamation type')                                    
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=hw[0],
                                  padding=int(hw[0]*0.125),
                                  padding_mode='reflect')])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=hw[0],
                                  padding=int(hw[0]*0.125),
                                  padding_mode='reflect'),
                                RandAugmentMC(n=2, m=10)])
        if resize:
            self.weak.transforms.insert(0, resize)
            self.strong.transforms.insert(0, resize)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.mode = mode

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.mode == 'fixmatch_strong2':
            strong2 = self.strong(x)
            return self.normalize(strong), self.normalize(strong2), self.normalize(weak)
        elif self.mode == 'strong2':
            strong2 = self.strong(x)
            return self.normalize(strong), self.normalize(strong2),
        elif self.mode == 'fixmatch':
            return self.normalize(strong), self.normalize(weak)
        elif self.mode == 'strong':
            return self.normalize(strong)
        else:
            raise error('Invalid strong augmentation')

