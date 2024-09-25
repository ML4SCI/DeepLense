# adapted from
#     https://github.com/facebookresearch/dino/blob/main/main_dino.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# DINO Loss
class DDINOLossL1(nn.Module):
    '''
    Implements the DINO loss function
    as mentioned in the original work
    
    params:
        
            
    return:
        DINO loss 
    '''
    
    def __init__(
            self,
            output_dim,
            num_crops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9
        ):
        super(DDINOLossL1, self).__init__()
        self.num_crops = num_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))
        self.register_buffer("center_grid", torch.zeros(1, output_dim))
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
        ) -> torch.FloatTensor:

        student_cls_out, student_region_out, student_fea, student_npatch = student_output
        teacher_cls_out, teacher_region_out, teacher_fea, teacher_npatch = teacher_output

        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_cls_out - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        teacher_region = F.softmax((teacher_region_out - self.center_grid) / teacher_temp, dim=-1)
        teacher_region = teacher_region.detach().chunk(2)
        
        teacher_fea = teacher_fea.chunk(2)
        
        N = teacher_npatch[0] # num of patches in the first view
        B = teacher_region[0].shape[0]//N # batch size,
        # sharpen
        student_out = (student_cls_out / self.student_temp).chunk(self.num_crops)

        student_region = student_region_out / self.student_temp
        student_split_size = [student_npatch[0]] * 2 + [student_npatch[1]] * (self.num_crops - 2) 
        student_split_size_bs = [i * B for i in student_split_size]
        
        student_region = torch.split(student_region, student_split_size_bs, dim=0)
        student_fea = torch.split(student_fea, student_split_size_bs, dim=0)

        

        

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx, s_out in enumerate(student_out):
                if s_idx == t_idx: # same view
                    continue
                loss = 0.5 * torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1)
                loss += 0.005 * torch.mean((torch.abs(t_out - s_out)))

                # region level prediction loss
                student_region_cur, student_fea_cur = student_region[s_idx].view(B, student_split_size[s_idx], -1), student_fea[s_idx].view(B, student_split_size[s_idx], -1)  # B x T_s x K, B x T_s x P
                teacher_region_cur, teacher_fea_cur = teacher_region[t_idx].view(B, N, -1), teacher_fea[t_idx].view(B, N, -1)  # B x T_t x K, B x T_t x P, 

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(student_fea_cur, p=2, dim=-1) , F.normalize(teacher_fea_cur, p=2, dim=-1) .permute(0, 2, 1)) # B x T_s x T_t
                region_sim_ind = region_sim_matrix.max(dim=2)[1] # B x T_s; collect the argmax index in teacher for a given student feature
                
                teacher_indexed_region = torch.gather(teacher_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, teacher_region_cur.size(2))) # B x T_s x K (index matrix: B, T_s, 1)

                loss += 0.5 * torch.sum(- teacher_indexed_region * F.log_softmax(student_region_cur, dim=-1), dim=[-1]).mean(-1)   # B x T_s x K --> B 

                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        self.update_center(teacher_cls_out, teacher_region_out)

        return total_loss
        
    @torch.no_grad()
    def update_center(self, teacher_output, teacher_grid_output):
        """
        Update center used for teacher output.
        """
        # view level center update
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # region level center update
        batch_grid_center = torch.mean(teacher_grid_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)

