from __future__ import print_function

import torch
import torch.nn as nn

class FeatureCache(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, dim, K, T=0.07):
        super(FeatureCache, self).__init__()
        self.K = K
        self.T = T

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("label_queue", torch.zeros(K))
        self.label_queue -= 1
        self.register_buffer("cr_mask_queue", torch.zeros(K, dtype=torch.long))

    
    def clear(self):
        self.queue.fill_(0)
        self.queue_ptr.fill_(0)
        self.label_queue.fill_(-1)
        self.cr_mask_queue.fill_(0)

    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, cr_masks):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            self.queue[:, ptr:] = keys.T[:, :self.K-ptr]
            self.label_queue[ptr:] = labels[:self.K-ptr]
            self.queue[:, :batch_size-self.K+ptr] = keys.T[:, self.K-ptr:]
            self.label_queue[:batch_size-self.K+ptr] = labels[self.K-ptr:]
            if cr_masks is not None:
                self.cr_mask_queue[ptr:] = cr_masks[:self.K-ptr]
                self.cr_mask_queue[:batch_size-self.K+ptr] = cr_masks[self.K-ptr:]
        else: 
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.label_queue[ptr:ptr + batch_size] = labels
            if cr_masks is not None:
                self.cr_mask_queue[ptr:ptr + batch_size] = cr_masks
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, q, k, label, cr_mask=None):
        queue_ = self.queue.clone().detach()  # (dim, K)
        label_ = self.label_queue.clone().detach()

        queue_enqueue = torch.cat([k.T, queue_], dim=1)  # shape: (dim, (bsz+qsz))
        label_enqueue = torch.cat([label, label_], dim=0)  # bsz+qsz

        # compute logits
        logits = torch.matmul(q, queue_enqueue)  # shape: (bsz, (bsz+qsz))
        logits = torch.div(logits, self.T)

        mask = label_enqueue.eq(label.view(-1, 1)).long()  # (bsz, (bsz+qsz))
        if cr_mask is not None:
            cr_mask_ = self.cr_mask_queue.clone().detach()
            cr_mask_enqueue = torch.cat([cr_mask, cr_mask_], dim=0)
            mask *= cr_mask_enqueue  # mask-out pseudo whose max logits less than threshold

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
            self._dequeue_and_enqueue(k, label, cr_mask)
        return loss.mean()
    
