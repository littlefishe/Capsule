import os
import sys
import copy
import logging
import math
import time
import inspect
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from utils import datasets
from models import create_model_instance
from utils.gpu_mem_track import MemTracker


class ClientTrainer():
    def __init__(self, process_id, 
                 select_idxs,
                 args, 
                 server_sync_barrier, client_sync_barrier, aggr_sync_barrier,
                 distr_model_list, bsz_list, embs, ema_embs, grad_list,
                 status_list, upload_cost_list, train_time_list,
                 lock, train_data_idxes,
                 aggr_model, aggr_device=torch.device("cuda:4")) -> None:
        self.args = args
        self.process_id = process_id
        self.select_idxs = select_idxs
        self.server_sync_barrier = server_sync_barrier
        self.client_sync_barrier = client_sync_barrier 
        self.aggr_sync_barrier = aggr_sync_barrier
        self.distr_model_list = distr_model_list
        self.bsz_list = bsz_list
        self.embs = embs
        self.ema_embs = ema_embs
        self.grad = grad_list
        self.status_list = status_list
        self.upload_cost_list = upload_cost_list
        self.train_time_list = train_time_list
        self.lock = lock
        self.train_data_idxes = train_data_idxes
        self.aggr_model = aggr_model
        self.aggr_device = aggr_device

        # init logger
        RESULT_PATH = os.getcwd() + '/client_logs/' + args.expname + '/'

        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH, exist_ok=True)

        self.logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
        self.logger.setLevel(logging.INFO)

        filename = RESULT_PATH + os.path.basename(__file__).split('.')[0] + '_'+ str(int(process_id)) +'.log'
        fileHandler = logging.FileHandler(filename=filename, mode='w')
        formatter = logging.Formatter("%(message)s")
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        
        self.logger.info("client_id:{}".format(process_id))
        self.logger.info("start")

        # init device
        visible_device = process_id % 4
        self.device = torch.device("cuda:%d" % visible_device if torch.cuda.is_available() else "cpu")
        
        utrain_dataset, _ = datasets.load_datasets(args.dataset_type, transform_type=args.utransform_type)
        unlabeled_idxes = train_data_idxes
        num_expand_x = math.ceil(args.batch_size * 50 / len(unlabeled_idxes))
        unlabeled_idxes = np.hstack([unlabeled_idxes for _ in range(num_expand_x)])
        np.random.shuffle(unlabeled_idxes)
        unlabeled_idxes = list(unlabeled_idxes)
        self.local_loader = datasets.create_dataloaders(utrain_dataset, batch_size=args.batch_size, selected_idxs=unlabeled_idxes, drop_last=True)

        self.logger.info("init loader: %d samples" % len(unlabeled_idxes))

        self.encoder = create_model_instance(args.model_type + 'Client')
        self.encoder.to(self.device)
        self.t_encoder = copy.deepcopy(self.encoder)
        for p in self.t_encoder.parameters():
            p.requires_grad_(False)
        local_paras = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).detach()
        self.upload_model_cost = local_paras.nelement()
        del local_paras
        self.logger.info("complete init")


    def split_train(self, local_optim):
        for iter_idx in range(self.args.local_steps):
            self.server_sync_barrier.wait()
            
            batch_size = self.bsz_list[self.process_id]

            (inputs, ema_inputs), targets = next(self.local_loader)
            inputs, ema_inputs, targets = inputs[:batch_size], ema_inputs[:batch_size], targets[:batch_size]
            inputs, ema_inputs, targets = inputs.to(self.device), ema_inputs.to(self.device), targets.to(self.device)
            

            embs = self.encoder(inputs)
            ema_embs = self.t_encoder(ema_inputs).detach()

            with self.lock:
                # self.embs_list[self.process_id] = embs.detach().cpu()
                # self.ema_embs_list[self.process_id] = ema_embs.cpu()

                self.embs.copy_(embs.detach())
                self.ema_embs.copy_(ema_embs)


            self.client_sync_barrier.wait()
            
            # --- waiting for server bp ---
            
            self.server_sync_barrier.wait()

            with self.lock:
                grads = self.grad.to(self.device)

            # client training

            local_optim.zero_grad()
            embs.backward(grads)
            local_optim.step()     

            self.client_sync_barrier.wait()

            # --- waiting for server processing... ---
        
        return
        
    
    def train(self):
        for round_idx in range(self.args.round):
            no_progress = round_idx / self.args.round
            lr = self.args.min_lr + 0.5 * (self.args.lr - self.args.min_lr) * (1 + np.cos(np.pi * no_progress))
            optimizer = optim.SGD(self.encoder.parameters(), 
                lr=lr, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)
            self.logger.info("round-{} lr: {}".format(round_idx, optimizer.param_groups[0]['lr']))
            
            # -- waiting for server training... --
            self.server_sync_barrier.wait()
            global_model_dict = self.distr_model_list.get()
            self.encoder.load_state_dict(global_model_dict[0])
            self.t_encoder.load_state_dict(global_model_dict[1])

            self.split_train(optimizer)

            with self.lock, torch.no_grad():
                if self.process_id in self.select_idxs:
                    for para in self.encoder.state_dict().keys():
                        temp_tensor = copy.deepcopy(self.encoder.state_dict()[para]).to(self.aggr_device)
                        self.aggr_model.state_dict()[para] += temp_tensor
                        del temp_tensor
            
            self.aggr_sync_barrier.wait()


            # -- waiting for the server to aggregate global model... --
       

def light_client_train(process_id, select_idxs, args, 
                       server_sync_barrier, client_sync_barrier, aggr_sync_barrier,
                       distr_model_list, bsz_list, embs, ema_embs, grad,
                       loss_list, upload_cost_list, train_time_list,
                       lock, train_data_idxes,
                       aggr_model, aggr_device):

    # disable print
    blockPrint()

    client_trainer = ClientTrainer(process_id, select_idxs, args, 
                                   server_sync_barrier, client_sync_barrier, aggr_sync_barrier,
                                   distr_model_list, bsz_list, embs, ema_embs, grad,
                                   loss_list, upload_cost_list, train_time_list,
                                   lock, train_data_idxes,
                                   aggr_model, aggr_device)
    client_trainer.train()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


