# adapted from
#     https://github.com/facebookresearch/dino/blob/main/main_dino.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple

# ---------------------------------------------------------------
# DINO Loss
class SWAVLoss(nn.Module):
    '''
    Implements the SWAV loss function
    as mentioned in the original work
    
    params:
        131072
            
    return:
        DINO loss 
    '''
    
    def __init__(
            self,
            crops_for_assign: int,
            num_crops: int,
            batch_size: int,
            temperature: float,
            head_output_dim: int,
            queue_length: int,
            epoch_queue_starts: int = 0,
            sinkhorn_iterations: int = 0,
            epsilon: float = 0.,
            device: str = "cuda"
        ):
        super(SWAVLoss, self).__init__()
        self.crops_for_assign = crops_for_assign
        self.num_crops = num_crops
        self.temperature = temperature
        self.batch_size = batch_size
        self.head_output_dim = head_output_dim
        self.queue = None
        self.queue_length = queue_length 
        self.epoch_queue_starts = epoch_queue_starts
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.use_the_queue = False
        self.device = device
        
    def forward(
            self,
            student_output: Tuple[torch.Tensor, ],
            epoch: int,
            prototypes_weight: torch.Tensor, # model.module.prototypes.weight.t()
            *args,
            **kwargs,
        ) -> torch.FloatTensor:
        
        total_loss = 0
        embedding, student_output = student_output
        embedding = embedding.detach()
        if self.queue_length > 0 and epoch >= self.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                                len(self.crops_for_assign),
                                self.queue_length,
                                self.head_output_dim,
                            ).to(self.device)
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = student_output[self.batch_size * crop_id: self.batch_size * (crop_id + 1)].detach()
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            prototypes_weight.t(),
                        ), out))
                    # fill the queue
                    self.queue[i, self.batch_size:] = self.queue[i, :-self.batch_size].clone()
                    self.queue[i, :self.batch_size] = embedding[crop_id * self.batch_size: (crop_id + 1) * self.batch_size]

                # get assignments
                q = self.sinkhorn(out)[-self.batch_size:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                x = student_output[self.batch_size * v: self.batch_size * (v + 1)] / self.temperature
                try:
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                except Exception as e:
                    print(epoch, i, crop_id)
                    print(v)
                    print(x)
                    print(len(student_output))
                    sys.exit(1)
            total_loss += subloss / (np.sum(self.num_crops) - 1)
        total_loss /= len(self.crops_for_assign)
        return total_loss

    def sinkhorn(self, x):
        Q = torch.exp(x / self.epsilon).t() 
        B = Q.shape[1] 
        K = Q.shape[0] 
    
        sum_Q = torch.mean(Q)
    
        for it in range(self.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
    
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
    
        Q *= B 
        return Q.t()

# --------------------------------------------------------------- 