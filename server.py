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
FLAGS = None

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
        FLAGS = SVHN.FLAGS
    elif args.dataset == "CIFAR10":
        FLAGS = CIFAR10.FLAGS
    elif args.dataset == "IMAGE100":
        FLAGS = IMAGE100.FLAGS
    else:
        raise ValueError("Unsupported dataset: %s" % args.dataset)
    
    FLAGS.expname = args.expname
    FLAGS.data_pattern = args.data_pattern
    if args.nlabeled > 0:
        FLAGS.labeled_num = args.nlabeled
    assert args.aworker <= args.nworker, "invalid active worker num"
    FLAGS.worker_num = args.nworker
    FLAGS.active_worker_num = args.aworker
    FLAGS.gpu = args.gpu


if __name__ == '__main__':
    print(time.strftime("%H:%M:%S"))
    arg_parse()
    # setup torch.multiprocessing
    ctx = multiprocessing.get_context('forkserver')

    server_service = ServerService(FLAGS, ctx)
    server_service.load_model()
    server_service.load_dataset()
    server_service.launch_clients(ctx, light_client_train)

    print('workers already launched!')
    
    global_step = [0]
    for round_idx in range(FLAGS.round):
        loss_x = server_service.sup_train(global_step, round_idx)

        server_service.model_dispatch()

        server_service.semi_sfl_train()
        
        loss_u = server_service.collect_params()
        
        test_loss, test_acc = server_service.test()
        print("Round [%d/%d] train loss %.4f, acc: %.4f" % (round_idx, FLAGS.round, loss_x+loss_u))

        server_service.system_control(round_idx, loss_x, loss_u)
    
    server_service.terminate()
    print(time.strftime("%H:%M:%S"))

