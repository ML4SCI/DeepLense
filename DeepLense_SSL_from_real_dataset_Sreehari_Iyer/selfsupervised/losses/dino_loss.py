# adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# DINO Loss
class DINOLoss(nn.Module):
    def __init__(
            self,
            output_dim,
            num_crops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9,
        ):
        super(DINOLoss, self).__init__()
        self.num_crops = num_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
        
    def forward(
            self,
            student_output: torch.Tensor,
            teacher_output: torch.Tensor,
            epoch: int,
            mask = None
        ) -> torch.FloatTensor:
        
        student_out = (student_output / self.student_temp).chunk(self.num_crops)

        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx, s_out in enumerate(student_out):
                if s_idx == t_idx: # same view
                    continue
                loss = torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
# --------------------------------------------------------------- 
