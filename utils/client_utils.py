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
    def __init__(self, *args, **kwargs) -> None:
        self.args = kwargs["args"]
        self.process_id = kwargs["process_id"]
        self.select_idxs = kwargs["select_idxs"]
        self.server_sync_barrier = kwargs["server_sync_barrier"]
        self.client_sync_barrier = kwargs["client_sync_barrier"]
        self.aggr_sync_barrier = kwargs["aggr_sync_barrier"]
        self.distr_model_list = kwargs["distr_model_list"]
        self.embs = kwargs["embs"]
        self.ema_embs = kwargs["ema_embs"]
        self.grad = kwargs["grad_list"]
        self.lock = kwargs["lock"]
        self.train_data_idxes = kwargs["train_data_idxes"]
        self.aggr_model = kwargs["aggr_model"]
        self.aggr_device = kwargs["aggr_device"]
        

        # init logger
        RESULT_PATH = os.getcwd() + '/client_logs/' + self.args.expname + '/'

        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH, exist_ok=True)

        self.logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
        self.logger.setLevel(logging.INFO)

        filename = RESULT_PATH + os.path.basename(__file__).split('.')[0] + '_'+ str(int(self.process_id)) +'.log'
        fileHandler = logging.FileHandler(filename=filename, mode='w')
        formatter = logging.Formatter("%(message)s")
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        
        self.logger.info("client_id:{}".format(self.process_id))
        self.logger.info("start")

        # init device
        visible_device = self.process_id % 4
        self.device = torch.device("cuda:%d" % visible_device if torch.cuda.is_available() else "cpu")
        
        utrain_dataset, _ = datasets.load_datasets(self.args.dataset_type, transform_type=self.args.utransform_type)
        unlabeled_idxes = self.train_data_idxes
        num_expand_x = math.ceil(self.args.batch_size * 50 / len(unlabeled_idxes))
        unlabeled_idxes = np.hstack([unlabeled_idxes for _ in range(num_expand_x)])
        np.random.shuffle(unlabeled_idxes)
        unlabeled_idxes = list(unlabeled_idxes)
        self.local_loader = datasets.create_dataloaders(utrain_dataset, batch_size=self.args.batch_size, selected_idxs=unlabeled_idxes, drop_last=True)

        self.logger.info("init loader: %d samples" % len(unlabeled_idxes))

        self.encoder = create_model_instance(self.args.model_type + 'Client')
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
            
            batch_size = self.args.batch_size

            (inputs, ema_inputs), targets = next(self.local_loader)
            inputs, ema_inputs, targets = inputs[:batch_size], ema_inputs[:batch_size], targets[:batch_size]
            inputs, ema_inputs, targets = inputs.to(self.device), ema_inputs.to(self.device), targets.to(self.device)
            

            embs = self.encoder(inputs)
            ema_embs = self.t_encoder(ema_inputs).detach()

            with self.lock:
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
    
    def noop_train(self):
        for iter_idx in range(self.args.local_steps):
            self.server_sync_barrier.wait()       

            self.client_sync_barrier.wait()
            
            # --- waiting for server bp ---
            
            self.server_sync_barrier.wait()

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

            if self.process_id in self.select_idxs:
                self.split_train(optimizer)
            else:
                self.noop_train()

            with self.lock, torch.no_grad():
                if self.process_id in self.select_idxs:
                    for para in self.encoder.state_dict().keys():
                        temp_tensor = copy.deepcopy(self.encoder.state_dict()[para]).to(self.aggr_device)
                        self.aggr_model.state_dict()[para] += temp_tensor
                        del temp_tensor
            
            self.aggr_sync_barrier.wait()

            # -- waiting for the server to aggregate global model... --
       

def client_train_warpper(*args, **kwargs):

    # disable print
    blockPrint()

    client_trainer = ClientTrainer(*args, **kwargs)
    client_trainer.train()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


