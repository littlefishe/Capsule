import os
import argparse
import time
import torch
import torch.multiprocessing as multiprocessing
from utils.server_utils import *
from utils.client_utils import *
from utils.losses import *
from utils.gpu_mem_track import MemTracker
from FLAGS import SVHN, CIFAR10, IMAGE100, STL10


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')
LOG_DIR = "./server_logs"

def arg_parse():
    #init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset', type=str, default='svhn')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--nlabeled', type=int, default=0)
    parser.add_argument('--clabeled', type=int, default=0)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--nworker', type=int, default=30)
    parser.add_argument('--aworker', type=int, default=10)
    parser.add_argument('--round', type=int, default=1000)
    parser.add_argument('--log_to_file', action="store_true", default=False)
    args = parser.parse_args()
    
    if args.dataset == "svhn":
        kargs = SVHN.FLAGS
    elif args.dataset == "cifar10":
        kargs = CIFAR10.FLAGS
    elif args.dataset == "image100":
        kargs = IMAGE100.FLAGS
    elif args.dataset == 'stl10':
        kargs = STL10.FLAGS
    else:
        raise ValueError("Unsupported dataset: %s" % args.dataset)
    
    if args.log_to_file:
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        logname = os.path.join(LOG_DIR, "%s-%d-%s%s" % (args.dataset, args.nlabeled if args.nlabeled > 0 else kargs.labeled_num, args.expname, ".log"))
        if os.path.exists(logname):
            ctn = input("file %s exists, continue? (Y or N): " % logname)
            if ctn != "Y":
                exit(0)
        sys.stdout = open(logname, "w")
    
    kargs.expname = args.expname
    kargs.data_pattern = args.data_pattern
    kargs.round = args.round
    if args.nlabeled > 0:
        kargs.labeled_num = args.nlabeled
    if args.clabeled > 0:
        kargs.client_labels = args.clabeled
    assert args.aworker <= args.nworker, "invalid active worker num"
    kargs.worker_num = args.nworker
    kargs.active_worker_num = args.aworker
    kargs.gpu = args.gpu
    return kargs


if __name__ == '__main__':
    args = arg_parse()
    print(time.strftime("%H:%M:%S"))
    print("worker %d | active %d" % (args.worker_num, args.active_worker_num))
    print("[HP] alpha=%.1f, beta=%.1f, tmpr=%.2f, thr=%.2f, gamma=%.2f, qsz=%d" % 
          (args.alpha, args.beta, args.temperature, args.threshold, args.ema_decay, args.queue_size))
    # setup torch.multiprocessing
    ctx = multiprocessing.get_context('forkserver')

    server_service = ServerService(args, ctx)
    server_service.load_model()
    server_service.load_dataset()
    server_service.launch_clients(ctx, client_train_warpper)

    print('workers already launched!')
    
    global_step = [0]
    durs = [0] * 5
    detdurs = [0] * 5
    for round_idx in range(args.round):
        loss_x = server_service.sup_train(global_step, round_idx) 
        if round_idx >= args.pre_round:
            server_service.model_dispatch()
            loss_u, detdurs = server_service.semi_sfl_train()
            server_service.collect_params()
        else:
            loss_u = 0
        
        ttest_loss, tacc = server_service.test_tea()
        print("Round [%d/%d] train loss %.4f" % (round_idx, args.round, loss_x+loss_u), end='')
        print(" | test loss %.4f, acc %.4f" % (ttest_loss, tacc), end='')
       
        print(" | lr %.4f, xloss %.4f, uloss %.4f, xstep %d, ustep %d" % (server_service.lr, loss_x, loss_u, 
                                                                       server_service.global_steps, server_service.args.local_steps), end='')

        server_service.system_control(round_idx, loss_x, loss_u)

        print(flush=True)

    server_service.terminate()
    print(time.strftime("%H:%M:%S"))
    sys.stdout = sys.__stdout__

