import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import TrainSSL
from selfsupervised.models import MultiCropWrapper, iBOTHead
from selfsupervised.losses import iBOTLoss
from selfsupervised.utils import cosine_scheduler
from typing import List, Dict, Union, Optional, Tuple

class TrainIBOT(TrainSSL):
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            logger,
            student_backbone: nn.Module,
            teacher_backbone: nn.Module,
            data_path,
            train_test_indices,
            data_augmentation_transforms,
            eval_transforms,
            pred_size: int, # = patch_size in case of ViT
            num_classes: int,
            train_val_split: Tuple[float, ],
            batch_size: int,
            # physical_batch_size: int,
            embed_dim: int,
            num_local_crops: int,
            scheduler_warmup_epochs: int,
            warmup_teacher_temp: float,
            warmup_teacher_patch_temp: float,
            teacher_temp: float,
            teacher_patch_temp: float,
            warmup_teacher_temp_epochs: int,
            lambda1: int,
            lambda2: int,
            momentum_teacher: float,
            num_epochs: int,
            head_output_dim: int,
            head_hidden_dim: int,
            head_bottleneck_dim: int,
            patch_size,
            pred_ratio,
            pred_ratio_var,
            pred_aspect_ratio,
            pred_shape,
            pred_start_epoch,
            shared_head,
            shared_head_teacher,
            patch_out_dim,
            clip_grad_magnitude: float = 0.3,
            restore_from_ckpt: bool = False,
            restore_ckpt_path: Optional[str] = None,
            lr: float = 1e-3,
            final_lr: Optional[float] = None,
            weight_decay: float = 5e-3,
            final_weight_decay: Optional[float] = None,
            use_dense_prediction: bool = False,
            head_use_bn: bool = False,
            head_norm_last_layer: bool = True,
            head_nlayers: int = 3,
            optimizer: Union["AdamW, Adam, SGD, LARS"] = "AdamW",
            log_freq: int = 5,
            device: str = "cuda",
            use_mixed_precision: bool = "True",
            freeze_last_layer: int = 1,
            knn_neighbours: int = 5,
            use_L1: bool = False,
            **kwargs,
            
        ):
        assert data_augmentation_transforms is not None
        assert eval_transforms is not None
        super().__init__(
            output_dir=output_dir,
            expt_name=expt_name,
            restore_from_ckpt=restore_from_ckpt,
            restore_ckpt_path=restore_ckpt_path,
            data_path=data_path,
            train_test_indices=train_test_indices,
            data_augmentation_transforms=data_augmentation_transforms,
            eval_transforms=eval_transforms,
            num_classes=num_classes,
            train_val_split = train_val_split,
            batch_size=batch_size,
            # physical_batch_size=physical_batch_size,
            num_epochs = num_epochs,
            log_freq = log_freq,
            device=device,
            logger=logger,
            use_mixed_precision=use_mixed_precision,
            knn_neighbours = knn_neighbours,
            use_dense_prediction = use_dense_prediction,
            masked_loader = True,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch,
            **kwargs
        )
        self.logger.info("Initializing iBOT training")

        self.num_local_crops = num_local_crops
        self.clip_grad_magnitude = clip_grad_magnitude
        self.freeze_last_layer = freeze_last_layer

        self.pred_size = patch_size 
        
        student_head_dense, teacher_head_dense = None, None
        if self.use_dense_prediction:
            student_head_dense = iBOTHead(
                in_dim = embed_dim, 
                out_dim = head_output_dim, 
                hidden_dim = head_hidden_dim,
                bottleneck_dim = head_bottleneck_dim,
                use_bn = head_use_bn, 
                norm_last_layer = head_norm_last_layer, 
                nlayers = head_nlayers,
                shared_head=shared_head_teacher,
            )
            teacher_head_dense = iBOTHead(
                in_dim = embed_dim, 
                out_dim = head_output_dim, 
                hidden_dim = head_hidden_dim,
                bottleneck_dim = head_bottleneck_dim,
                use_bn = head_use_bn, 
                norm_last_layer = True, 
                nlayers = head_nlayers,
                shared_head=shared_head_teacher,
            )
        # student and teacher network
        self.student = MultiCropWrapper(
                    backbone = student_backbone,
                    head = iBOTHead(
                        in_dim = embed_dim, 
                        out_dim = head_output_dim, 
                        hidden_dim = head_hidden_dim,
                        bottleneck_dim = head_bottleneck_dim,
                        use_bn = head_use_bn, 
                        norm_last_layer = head_norm_last_layer, 
                        nlayers = head_nlayers,
                        shared_head=shared_head,
                    ),
                    use_dense_prediction=self.use_dense_prediction,
                    head_dense = student_head_dense,
                )
        self.teacher = MultiCropWrapper(
                    backbone = teacher_backbone,
                    head = iBOTHead(
                        in_dim = embed_dim, 
                        out_dim = head_output_dim, 
                        hidden_dim = head_hidden_dim,
                        bottleneck_dim = head_bottleneck_dim,
                        use_bn = head_use_bn, 
                        norm_last_layer = True, 
                        nlayers = head_nlayers,
                        shared_head=shared_head_teacher,
                    ),
                    use_dense_prediction=self.use_dense_prediction,
                    head_dense = teacher_head_dense,
                )
        self.student, self.teacher = self.student.to(device), self.teacher.to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False


        self.logger.info("Built Student and Teacher networks.\nBoth are initialized with same parameters")

        # loss
        same_dim = shared_head or shared_head_teacher
        self.criterion = iBOTLoss(
                            out_dim = head_output_dim,
                            patch_out_dim = patch_out_dim,
                            ngcrops=2, nlcrops=self.num_local_crops,
                            warmup_teacher_temp = warmup_teacher_temp,
                            teacher_temp = teacher_temp,
                            warmup_teacher_temp2 = warmup_teacher_patch_temp,
                            teacher_temp2 = teacher_patch_temp,
                            warmup_teacher_temp_epochs = warmup_teacher_temp_epochs,
                            nepochs = self.epochs, 
                            lambda1=lambda1, lambda2=lambda2,
                            mim_start_epoch=pred_start_epoch
                        ).to(self.device)
        # optimizer
        # lr is assigned by scheduler
        try:
            if optimizer == "AdamW":
                self.optimizer = torch.optim.AdamW(self.student.parameters())
            elif optimizer == "Adam":
                self.optimizer = torch.optim.Adam(self.student.parameters()) 
            elif optimizer == "SGD":
                self.optimizer = torch.optim.SGD(self.student.parameters(), lr=0., momentum=0.9) 
            elif optimizer == "LARS":
                raise NotImplementedError(f"{optimizer} not implemented")
            else:
                raise NotImplementedError(f"{optimizer} not implemented")
        except NotImplementedError as error:
            self.logger.error(error)
            raise error

        self.logger.info(f"Initialized {optimizer} Optimizer")

        # scheduler
        self.lr_schedule = cosine_scheduler(
            init_val = lr, 
            final_val = final_lr,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
            warmup_epochs = scheduler_warmup_epochs,
        )
        self.wd_schedule = cosine_scheduler(
            init_val = weight_decay,
            final_val = final_weight_decay,
            epochs = self.epochs, 
            start_warmup_value = weight_decay,
            steps_per_epoch = self.steps_per_epoch,
        )
        self.momentum_schedule = cosine_scheduler(
            init_val = momentum_teacher, 
            final_val = 1,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
        )
        self.logger.info(f"Initialized schedulers")

        self._init_state()

    def forward_teacher(self, img, *args, **kwargs):
        # print(img[0].shape)
        return self.teacher(img[:2], return_backbone_feat=False) 

    def forward_student(self, img, msk=None, *args, **kwargs):
        img, msk = img
        student_output = self.student(img[:2], mask=msk[:2], return_backbone_feat=False)
        # get local views
        restore_masked_im_modeling = self.student.backbone.masked_im_modeling
        self.student.backbone.masked_im_modeling = False
        student_local_cls = self.student(img[2:], return_backbone_feat=False)[0] if len(img) > 2 else None
        self.student.backbone.masked_im_modeling = restore_masked_im_modeling
        return (student_output, student_local_cls)
        

    def set_lr_and_wd(self, current_step):
        lr, wd = self.lr_schedule[current_step], self.wd_schedule[current_step]
        for param_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
            if param_idx == 0:  
                param_group["weight_decay"] = wd

    def compute_loss_epoch(self, student_output, teacher_output, mask):
        return self.criterion(student_output, teacher_output, self.state["current_epoch"], mask)

    def process_grads_before_step(self, epoch, *args, **kwargs): 
        if self.clip_grad_magnitude != 0:
            if self.fp16_scaler is not None:
                self.fp16_scaler.unscale_(self.optimizer)
            for param in self.student.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    clip_coef = self.clip_grad_magnitude / (param_norm + 1e-6)
                    if clip_coef < 1:
                        param.grad.data.mul_(clip_coef)
        if self.freeze_last_layer != 0 and epoch < self.freeze_last_layer:
            self._cancel_gradients_last_layer(epoch, self.student, self.freeze_last_layer)
        return
      

    def _cancel_gradients_last_layer(self, epoch, model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None
        return

    @torch.no_grad()
    def update_teacher(self, current_step):
        teacher_momentum = self.momentum_schedule[current_step]  # momentum parameter
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(teacher_momentum).add_((1 - teacher_momentum) * student_param.detach().data)
    

        






        
