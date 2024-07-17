import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import datasets

def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * \
        ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes


def dirichlet_partition(dataset_type:str, alpha: float, worker_num: int, nclasses: int):
    partition_sizes = []
    filepath = '../data_partition_%d/%s_dir%s.npy' % (worker_num, dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(worker_num):
            partition_sizes.append(np.random.dirichlet([alpha] * nclasses))
        partition_sizes = np.array(partition_sizes)
        partition_sizes /= partition_sizes.sum(axis=0)
        partition_sizes = partition_sizes.T
        np.save(filepath, partition_sizes)

    return partition_sizes


def partition_data(dataset_type, data_pattern, worker_num=10, nlabeled=0., transform_type='default', client_labels=0):
    if dataset_type == 'STL10':
        trainset, testset, unlabeledset = datasets.load_datasets(
            dataset_type, transform_type=transform_type)
        nlabeled = len(trainset)
    else:
        trainset, testset = datasets.load_datasets(dataset_type, transform_type=transform_type)
        unlabeledset = trainset

    labeled_ratio = nlabeled / len(trainset)

    if dataset_type == 'IMAGE100':
        nclasses = 100
    elif dataset_type in ['CIFAR10', 'SVHN', 'STL10']:
        nclasses = 10
    else:
        raise ValueError('Unsupported dataset type')
    
        
    if data_pattern == 0:  # iid  
        partition_sizes = np.ones((nclasses, worker_num)) / worker_num

    elif data_pattern == 1:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, nclasses)

    elif data_pattern == 2:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, nclasses)

    elif data_pattern == 3:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, nclasses)

    elif data_pattern == 4:  # dir-0.05
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.05, worker_num, nclasses)

    elif data_pattern == 5:  # dir-0.01
        print('Dirichlet partition 0.01')
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, nclasses)
        

    if dataset_type != 'STL10':
        if client_labels:
            partition_sizes = np.concatenate([
                np.ones((nclasses, 1)) * (nlabeled - client_labels) / len(trainset) , 
                np.ones((nclasses, worker_num)) * (client_labels / worker_num) / len(trainset),
                partition_sizes * (1.-labeled_ratio)], axis=1)
        else:
            partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                partition_sizes * (1.-labeled_ratio), 
                                ], axis=1)

    partition = datasets.LabelwisePartitioner(
        unlabeledset, partition_sizes=partition_sizes)
    
    if dataset_type == 'STL10' and client_labels > 0:
        partition_sizes = np.concatenate([
                                    np.ones((nclasses, 1)) * (nlabeled-client_labels) / nlabeled, 
                                    partition_sizes * client_labels / nlabeled], axis=1)
        labeled_partition = datasets.LabelwisePartitioner(
            trainset, partition_sizes=partition_sizes)
    else:
        labeled_partition = None


    return trainset, testset, partition, unlabeledset, labeled_partition
