from __future__ import print_function

import torch
import torch.nn as nn

 
class FeatureCache(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, dim, K, T=0.07, F=10):
        super(FeatureCache, self).__init__()
        self.K = K
        self.T = T

        self.register_buffer("squeue", torch.randn(dim, K))
        self.register_buffer("uqueue", torch.randn(dim, K))
        self.squeue = nn.functional.normalize(self.squeue, dim=0)
        self.uqueue = nn.functional.normalize(self.uqueue, dim=0)
        self.register_buffer("squeue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("uqueue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("label_queue", torch.zeros(K))
        self.register_buffer("ulabel_queue", torch.zeros(K))
        self.slabel_queue -= 1
        self.ulabel_queue -= 1 
        self.register_buffer("mask_queue", torch.zeros(K, dtype=torch.long))
        self.register_buffer("smask", torch.ones(K, dtype=torch.long))

        self.F = F
    
    def clear(self):
        self.squeue.fill_(0)
        self.uqueue.fill_(0)
        self.squeue_ptr.fill_(0)
        self.uqueue_ptr.fill_(0)
        self.slabel_queue.fill_(-1)
        self.ulabel_queue.fill_(-1)
        self.mask_queue.fill_(0)
        self.smask.fill_(1)

    
    @torch.no_grad()
    def _udequeue_and_enqueue(self, keys, labels, masks):
        batch_size = keys.shape[0]
        ptr = int(self.uqueue_ptr)
        if ptr + batch_size > self.K:
            self.uqueue[:, ptr:] = keys.T[:, :self.K-ptr]
            self.ulabel_queue[ptr:] = labels[:self.K-ptr]
            self.uqueue[:, :batch_size-self.K+ptr] = keys.T[:, self.K-ptr:]
            self.ulabel_queue[:batch_size-self.K+ptr] = labels[self.K-ptr:]
            self.mask_queue[ptr:] = masks[:self.K-ptr]
            self.mask_queue[:batch_size-self.K+ptr] = masks[self.K-ptr:]
        else: 
            self.uqueue[:, ptr:ptr + batch_size] = keys.T
            self.ulabel_queue[ptr:ptr + batch_size] = labels
            self.mask_queue[ptr:ptr + batch_size] = masks
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.uqueue_ptr[0] = ptr
    
    @torch.no_grad()
    def _sdequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.squeue_ptr)
        if ptr + batch_size > self.K:
            self.squeue[:, ptr:] = keys.T[:, :self.K-ptr]
            self.slabel_queue[ptr:] = labels[:self.K-ptr]
            self.squeue[:, :batch_size-self.K+ptr] = keys.T[:, self.K-ptr:]
            self.slabel_queue[:batch_size-self.K+ptr] = labels[self.K-ptr:]
        else: 
            self.squeue[:, ptr:ptr + batch_size] = keys.T
            self.slabel_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.squeue_ptr[0] = ptr


    def forward(self, q, k, label, sup=False, cr_mask=None, step=0):
        if sup:
            queue_ = self.squeue.clone().detach()  # (dim, K)
            label_ = self.slabel_queue.clone().detach()
        else:
            queue_ = torch.cat(self.uqueue, self.squeue).clone().detach()
            label_ = torch.cat(self.ulabel_queue, self.slabel_queue).clone().detach()

        queue_enqueue = torch.cat([k.T, queue_], dim=1)  # shape: (dim, (bsz+qsz))
        label_enqueue = torch.cat([label, label_], dim=0)  # bsz+qsz

        # compute logits
        logits = torch.matmul(q, queue_enqueue)  # shape: (bsz, (bsz+qsz))
        logits = torch.div(logits, self.T)

        mask = label_enqueue.eq(label.view(-1, 1)).long()  # (bsz, (bsz+qsz))
        if cr_mask is not None:
            mask_ = self.mask_queue.clone().detach()
            mask_enqueue = torch.cat([cr_mask, mask_, self.smask], dim=0)
            mask *= mask_enqueue  # mask-out pseudo whose max logits less than threshold

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum>0, mask_sum, 1)  # prevent division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = -mean_log_prob_pos

        # update memory
        if self.K > 0:
            self._udequeue_and_enqueue(k, label, cr_mask)
            if sup or step % self.F == 0:
                self._sdequeue_and_enqueue(k, label)
        return loss.mean()
    
    
