import os
import argparse
import time
import torch
import torch.multiprocessing as multiprocessing
from utils.server_utils import *
from utils.client_utils import *
from utils.losses import *
from utils.gpu_mem_track import MemTracker
from FLAGS import SVHN, CIFAR10, IMAGE100


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')

def arg_parse():
    #init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset', type=str, default='SVHN')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--nlabeled', type=int, default=0)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--nworker', type=int, default=10)
    parser.add_argument('--aworker', type=int, default=5)
    args = parser.parse_args()
    
    if args.dataset == "SVHN":
        kargs = SVHN.FLAGS
    elif args.dataset == "CIFAR10":
        kargs = CIFAR10.FLAGS
    elif args.dataset == "IMAGE100":
        kargs = IMAGE100.FLAGS
    else:
        raise ValueError("Unsupported dataset: %s" % args.dataset)
    
    kargs.expname = args.expname
    kargs.data_pattern = args.data_pattern
    if args.nlabeled > 0:
        kargs.labeled_num = args.nlabeled
    assert args.aworker <= args.nworker, "invalid active worker num"
    kargs.worker_num = args.nworker
    kargs.active_worker_num = args.aworker
    kargs.gpu = args.gpu
    return kargs


if __name__ == '__main__':
    print(time.strftime("%H:%M:%S"))
    args = arg_parse()
    # setup torch.multiprocessing
    ctx = multiprocessing.get_context('forkserver')

    server_service = ServerService(args, ctx)
    server_service.load_model()
    server_service.load_dataset()
    server_service.launch_clients(ctx, client_train_warpper)

    print('workers already launched!')
    
    global_step = [0]
    for round_idx in range(args.round):
        loss_x = server_service.sup_train(global_step, round_idx)

        server_service.model_dispatch()

        loss_u = server_service.semi_sfl_train()
        
        server_service.collect_params()
        
        test_loss, test_acc = server_service.test()
        print("Round [%d/%d] train loss %.4f, acc: %.4f" % (round_idx, args.round, loss_x+loss_u, test_acc))

        server_service.system_control(round_idx, loss_x, loss_u)
    
    server_service.terminate()
    print(time.strftime("%H:%M:%S"))

