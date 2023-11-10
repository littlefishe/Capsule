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
    filepath = '../data_partition/%s_dir%s.npy' % (dataset_type, alpha)
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


def partition_data(dataset_type, data_pattern, worker_num=10, n_labeled=0., transform_type='default'):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type, transform_type=transform_type)
    labeled_ratio = n_labeled / len(train_dataset)
    if dataset_type in ['CIFAR100', 'IMAGE100']:
        nclasses = 100
    elif dataset_type in ['CIFAR10', 'SVHN']:
        nclasses = 10
    else:
        raise ValueError('Unsupported dataset type')
    
    if data_pattern == 0:  # iid  

        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    np.ones((nclasses, worker_num)) * (1.-labeled_ratio)/worker_num, 
                                    ], axis=1)

    elif data_pattern == 1:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, nclasses)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    partition_sizes * (1.-labeled_ratio), 
                                    ], axis=1)

    elif data_pattern == 2:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, nclasses)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    partition_sizes * (1.-labeled_ratio), 
                                    ], axis=1)

    elif data_pattern == 3:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, nclasses)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    partition_sizes * (1.-labeled_ratio), 
                                    ], axis=1)
    
    elif data_pattern == 4:  # dir-0.05
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.05, worker_num, nclasses)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    partition_sizes * (1.-labeled_ratio), 
                                    ], axis=1)
        
    elif data_pattern == 5:  # dir-0.01
        print('Dirichlet partition 0.01')
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, nclasses)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)) * labeled_ratio, 
                                    partition_sizes * (1.-labeled_ratio), 
                                    ], axis=1)
        
    elif data_pattern == 6:  # k=2  cifar-10
        partition_sizes = [ [1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                            ]
        partition_sizes = np.concatenate([partition_sizes] * (worker_num // 5), axis=1)
        partition_sizes = np.concatenate([np.ones((nclasses, 1)), partition_sizes], axis=1)
        partition_sizes[:, 0] *= labeled_ratio
        partition_sizes[:, 1:] *= (1-labeled_ratio) / (worker_num // 5)

    train_data_partition = datasets.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes)

    test_data_partition = datasets.LabelwisePartitioner(
        test_dataset, partition_sizes=np.ones((nclasses, 10)) * (1 / 10))

    return train_dataset, test_dataset, train_data_partition, test_data_partition
