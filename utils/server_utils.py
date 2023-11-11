import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import models
import math
from utils.losses import *
from utils import datasets
from utils.data import partition_data
from torch.cuda.amp import autocast, GradScaler


def update_average(ema_model, model, global_step, alpha=0.99, constant_ema=False):
    # Use the true average until the exponential average is more correct
    if not constant_ema:
        alpha = min(1 - 1 / (global_step + 1), alpha)

    state_dict_main = model.state_dict()
    state_dict_ema = ema_model.state_dict()
    for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
        assert k_main == k_ema, "state_dict names are different!"
        assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
        if 'num_batches_tracked' in k_ema:
            v_ema.copy_(v_main)
        else:
            v_ema.copy_(v_ema * alpha + (1. - alpha) * v_main)



class ServerService():
    def __init__(self, args, ctx) -> None:
        self.args = args
        self.device = torch.device("cuda:%s" % args.gpu if torch.cuda.is_available() else "cpu")
        self.aggr_device = torch.device("cuda:0")
        self.worker_num = args.worker_num
        self.active_num = args.active_worker_num

        self.server_sync_barrier = ctx.Barrier(args.worker_num + 1)
        self.client_sync_barrier = ctx.Barrier(args.worker_num + 1)
        self.aggr_sync_barrier = ctx.Barrier(args.worker_num + 1)
        self.aggr_model_list = ctx.Manager().list([0] * args.worker_num)
        self.status_list = ctx.Manager().list([0] * args.worker_num)
        self.upload_cost_list = ctx.Manager().list([0] * args.worker_num)
        self.train_time_list = ctx.Manager().list([0] * args.worker_num)

        self.select_idxs = ctx.Manager().list([0] * args.active_worker_num)

        self.embs_list = [torch.zeros(args.batch_size, args.emb_dim).to(self.device).share_memory_() for _ in range(args.worker_num)]
        self.ema_embs_list = [torch.zeros(args.batch_size, args.emb_dim).to(self.device).share_memory_() for _ in range(args.worker_num)]
        self.grad_list = [torch.zeros(args.batch_size, args.emb_dim).to(self.device).share_memory_() for _ in range(args.worker_num)]
        self.send_mq = [ctx.Manager().Queue(args.worker_num) for _ in range(args.worker_num)]
        self.lock = ctx.Manager().Lock()

        # self.scaler = GradScaler()
        self.batch_size = self.args.batch_size
        self.processes = []

        self.avg_loss_x = 0
        self.avg_loss_u = 0
        # self.sup_iters = self.args.sup_iters
        self.sup_iters = 80


    def load_model(self):
        self.classifier = models.create_model_instance(self.args.model_type + 'Server')
        self.encoder = models.create_model_instance(self.args.model_type + 'Client')
        self.proj_head = models.ProjectionHead(self.args.proj_type, dim_in=self.args.emb_dim, feat_dim=self.args.proj_dim)
        
        self.classifier.to(self.device)
        self.encoder.to(self.device)
        self.proj_head.to(self.device)

        self.aggr_classifier = models.create_model_instance(self.args.model_type + 'Server').to(self.aggr_device).share_memory()
        self.aggr_proj_head = models.ProjectionHead(self.args.proj_type, dim_in=self.args.emb_dim, feat_dim=self.args.proj_dim).to(self.aggr_device).share_memory()
        self.fcache = FeatureCache(self.args.proj_dim, self.args.uqueue_size, self.args.utemperature).to(self.device).share_memory()

        self.t_classifier = copy.deepcopy(self.classifier)
        self.t_encoder = copy.deepcopy(self.encoder)
        self.t_proj_head = copy.deepcopy(self.proj_head)

        print('init teacher')
        update_average(self.t_classifier, self.classifier, 0)
        update_average(self.t_encoder, self.encoder, 0)
        update_average(self.t_proj_head, self.proj_head, 0)
        for p in self.t_classifier.parameters():
            p.requires_grad_(False)
        for p in self.t_encoder.parameters():
            p.requires_grad_(False)
        for p in self.t_proj_head.parameters():
            p.requires_grad_(False)

        self.aggr_model = models.create_model_instance(self.args.model_type + 'Client').to(self.aggr_device).share_memory()
    
    def load_dataset(self):
        strain_dataset, test_dataset, train_data_partition, _ = \
            partition_data(self.args.dataset_type, self.args.data_pattern, self.args.worker_num, 
                            n_labeled=self.args.labeled_num, transform_type=self.args.stransform_type)

        print('data has been partitioned!')
        self.test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

        labeled_idx = train_data_partition.use(0)
        if self.args.expand:
            num_expand_x = math.ceil(self.args.sbatch_size * self.args.sup_iters / self.args.labeled_num)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
            np.random.shuffle(labeled_idx)
            labeled_idx = list(labeled_idx)
        self.strain_loader = datasets.create_dataloaders(strain_dataset, batch_size=self.args.sbatch_size, 
            selected_idxs=labeled_idx, drop_last=self.args.drop_last)

        # send init params
        self.train_data_idxes = [train_data_partition.use(worker_idx + 1) for worker_idx in range(self.args.worker_num)]


    def launch_clients(self, ctx, train_func):
        # create a list of processes
        
        for i in range(self.worker_num):
            p = ctx.Process(target=train_func,     
                            kwargs={
                                    "process_id":i,
                                    "select_idxs":self.select_idxs,
                                    "args":self.args, 
                                    "server_sync_barrier":self.server_sync_barrier, 
                                    "client_sync_barrier":self.client_sync_barrier,
                                    "aggr_sync_barrier":self.aggr_sync_barrier,
                                    "distr_model_list":self.send_mq[i], 
                                    "embs":self.embs_list[i], 
                                    "ema_embs":self.ema_embs_list[i], 
                                    "grad_list":self.grad_list[i],       
                                    "lock":self.lock,
                                    "train_data_idxes":self.train_data_idxes[i],
                                    "aggr_model":self.aggr_model,
                                    "aggr_device":self.aggr_device
                                }
                            )
            self.processes.append(p)

        # start all processes
        for p in self.processes:
            p.start()


    def sup_train(self, global_step, round_idx):
        no_progress = round_idx / self.args.round
        self.lr = self.args.min_lr + 0.5 * (self.args.lr - self.args.min_lr) * (1 + np.cos(np.pi * no_progress))
        s_optimizer = optim.SGD(itertools.chain(*[self.classifier.parameters(), self.encoder.parameters(), self.proj_head.parameters()]), 
            lr=self.lr, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)

        self.encoder.train()
        self.classifier.train()
        self.proj_head.train()
        loss_x = 0.
        local_steps = 0

        ema_decay = self.args.ema_decay ** (self.sup_iters / self.args.sup_iters)

        for i in range(self.args.sup_iters):
            (inputs, ema_inputs), targets = next(self.strain_loader)
            inputs, ema_inputs, targets = inputs.to(self.device), ema_inputs.to(self.device), targets.to(self.device)
            
            s_optimizer.zero_grad()
           
            embs = self.encoder(inputs)
            outputs = self.classifier(embs)
            feats = self.proj_head(embs)

            ce_loss = F.cross_entropy(outputs, targets)
        
            ema_embs = self.t_encoder(ema_inputs)
            keys = self.t_proj_head(ema_embs).detach()
            
            cr_loss = self.fcache(feats, keys, targets, True)

            loss = ce_loss + cr_loss
            
            s_optimizer.zero_grad()
            loss.backward()
            s_optimizer.step()

            loss_x += loss.data
            local_steps += 1

            global_step[0] += 1
            update_average(self.t_encoder, self.encoder, global_step[0], ema_decay)
            update_average(self.t_classifier, self.classifier, global_step[0], ema_decay)
            update_average(self.t_proj_head, self.proj_head, global_step[0], ema_decay)

        loss_x /= local_steps
        # self.fcache.clear()
        return loss_x.item()
    
    def semi_sfl_train(self):
        self.classifier.train()
        self.proj_head.train()
        self.t_classifier.eval()
        self.t_proj_head.eval()

        u_optimizer = optim.SGD(itertools.chain(*[self.classifier.parameters(), self.proj_head.parameters()]), 
                                lr=self.lr, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)
       
        loss_u = 0
        p_num = 0

        _select_idxs = np.random.choice(self.worker_num, size=self.active_num, replace=False)
        for i in range(len(_select_idxs)):
            self.select_idxs[i] = _select_idxs[i]

        for step_idx in range(self.args.local_steps):
            self.server_sync_barrier.wait()
                   
            # --- waiting for client split fp ... ---

            self.client_sync_barrier.wait()

            # --- server bp ----          
            
            self.xembs_list = torch.cat([embs.clone().detach() for i, embs in enumerate(self.embs_list) if i in self.select_idxs], dim=0)
            self.xema_embs_list = torch.cat([ema_embs.clone().detach() for i, ema_embs in enumerate(self.ema_embs_list) if i in self.select_idxs], dim=0)
            
            sum_bsz = self.batch_size * self.active_num

            u_optimizer.zero_grad()
            loss_u_i, p_i = self.server_update(step_idx)
            loss_u_i.backward()

            bsz_s = 0
            for i in range(self.worker_num):
                if i in self.select_idxs:
                    self.grad_list[i][:self.batch_size].copy_(self.xembs_list.grad[bsz_s: bsz_s + self.batch_size] * sum_bsz / self.batch_size)
                    bsz_s += self.batch_size
          
            u_optimizer.step()
            loss_u += loss_u_i.item()
            p_num += p_i

            self.server_sync_barrier.wait()

            # --- waiting for client bp ---

            self.client_sync_barrier.wait()
            self.fcache.clear()
        
        loss_u /= self.args.local_steps
        
        # --- waiting for client aggregation... ---

        self.aggr_sync_barrier.wait()

        return loss_u


    
    def model_dispatch(self):
        global_para = dict()
        t_para = dict()
        for para in self.encoder.state_dict().keys():
            global_para[para] = copy.deepcopy(self.encoder.state_dict()[para].cpu())
            t_para[para] = copy.deepcopy(self.t_encoder.state_dict()[para].cpu())

        # send model
        for i in range(self.worker_num):
            self.send_mq[i].put((global_para, t_para))

        # prepare for aggregation
        init_model_dict(self.aggr_model)
        self.aggr_classifier.load_state_dict(self.classifier.state_dict())
        self.aggr_proj_head.load_state_dict(self.proj_head.state_dict())

        self.server_sync_barrier.wait()

    
    def collect_params(self):
        with self.lock:
            # aggregate model paras and time in shared list
            self.encoder.load_state_dict(copy.deepcopy(self.aggr_model.state_dict()))
            for para in self.encoder.state_dict():
                if 'num_batches_tracked' in para:
                    self.encoder.state_dict()[para].copy_(torch.div(self.encoder.state_dict()[para], self.active_num).long())
                else:
                    self.encoder.state_dict()[para] /= self.active_num
        


    def server_update(self, step):
        # retain gradients for clients    
        self.xembs_list.requires_grad = True
        
        outputs = self.classifier(self.xembs_list)
        t_outputs = self.t_classifier(self.xema_embs_list).detach()
        
        t_logits = torch.softmax(t_outputs, dim=-1)
        t_cfd, t_preds = torch.max(t_logits, dim=-1)

        mask = t_cfd.ge(self.args.threshold).float()
        ce_loss = (F.cross_entropy(outputs, t_preds, reduction='none') * mask).mean() 

        feats = self.proj_head(self.xembs_list)
        key = self.t_proj_head(self.xema_embs_list).detach()

        cr_mask = t_cfd.ge(self.args.threshold).long()

        cr_loss = self.fcache(feats, key, t_preds, False, cr_mask, step)
        
        loss = ce_loss + cr_loss

        return loss, mask.sum()

    def system_control(self, round_idx, loss_x, loss_u):
        self.avg_loss_x += loss_x
        self.avg_loss_u += loss_u
        if round_idx == 0:
            self.last_avg_loss_x = loss_x
            self.last_avg_loss_u = loss_u
        elif round_idx % 10 == 0:
            self.avg_loss_x /= 10
            self.avg_loss_u /= 10
            delta_loss_x = self.last_avg_loss_x - self.avg_loss_x
            delta_loss_u = self.last_avg_loss_u - self.avg_loss_u
            
            # global updating frequency adaptation
            if self.args.iter_decay and round_idx in self.args.milestones:
                if abs(delta_loss_u) < self.args.pho1:
                    self.sup_iters /= 2
                elif abs(delta_loss_u) > self.args.pho2:
                    self.sup_iters *= 2

            self.last_avg_loss_x = self.avg_loss_x
            self.last_avg_loss_u = self.avg_loss_u
            self.avg_loss_x = 0
            self.avg_loss_u = 0


    def terminate(self):
        # wait for all processes to finish
        for p in self.processes:
            p.join()
        
    def test(self):
        self.encoder.eval()
        self.classifier.eval()
        if hasattr(self.test_loader, 'loader'):
            self.test_loader = self.test_loader.loader
        test_loss = 0.0
        test_accuracy = 0.0

        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:

                data, target = data.to(self.device), target.to(self.device)


                features = self.t_encoder(data)
                output = self.t_classifier(features)
            
                # sum up batch loss
                loss_func = nn.CrossEntropyLoss(reduction='sum') 
                test_loss += loss_func(output, target).item()
                
                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += batch_correct
                total += target.size(0)

        test_loss /= total
        test_accuracy = np.float(1.0 * correct / total)

        return test_loss, test_accuracy


def init_model_dict(model):
    with torch.no_grad():
        for para in model.state_dict().keys():
            if 'attention_bias_idxs' in para:
                continue
            model.state_dict()[para].fill_(0)
    return

