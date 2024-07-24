import os
import torch
import torch.nn as nn
from torchvision import transforms as Transforms
from torch.utils.data import DataLoader
from .base import TrainSSL
from models import MLPHead, MultiCropWrapper
from losses import SWAVLoss
from utils import cosine_scheduler
from typing import List, Dict, Union, Optional, Tuple
# from apex.parallel.LARC import LARC
from optimizer.LARC import LARC

class TrainSWAV(TrainSSL):
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            logger,
            data_path: str,
            train_test_indices: List[int],
            data_augmentation_transforms: Transforms,
            eval_transforms: Transforms,
            student_backbone,
            num_classes: int = 0, 
            train_val_split: Tuple[float] = (0.85, 0.15),
            num_epochs: int = 100,
            batch_size: int = None,
            lr: float = 5,
            final_lr: float = 0,
            weight_decay: float = 1e-6,
            final_weight_decay: float = 1e-6,
            scheduler_warmup_epochs: int = 10,
            start_warmup_lr: float = 0.0,
            num_crops: List[int] = [0,1],
            crops_for_assign: List[int] = [0, 1], 
            temperature: float = 0.1,
            sinkhorn_iterations: int = 3,
            epsilon: float = 0.05,
            num_prototypes: float = 3000,
            queue_length: int = 0,
            epoch_queue_starts: int = 15,
            freeze_prototypes_niters: int = 313,

            head_output_dim: int = 128,
            head_hidden_dim: int = 2048,
            head_use_bn: bool = False,
            head_norm_last_layer: bool = True,
            head_nlayers: int = 2,
            head_norm_layer: nn.Module = nn.BatchNorm1d,
            head_activation: nn.Module = nn.ReLU,
        
            restore_from_ckpt: bool = False,
            restore_ckpt_path: Optional[str] = None,
            
            optimizer: Union["SGD"] = "SGD",
            log_freq: int = 20,
            device: str = "cuda",
            use_mixed_precision: bool = "True",
            clip_grad_magnitude: int = 0.,
            knn_neighbours: int = 5,

            *args,
            **kwargs,
            
        ):
    #--------------------------------------------------------------------------------------------------------
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
            num_epochs = num_epochs,
            log_freq = log_freq,
            device=device,
            logger=logger,
            use_mixed_precision=use_mixed_precision,
            knn_neighbours = knn_neighbours,
            **kwargs
        )
        self.logger.info("Initializing SWAV training")
    #--------------------------------------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------------------------------------
        # initialize network, teacher is None in SWAV 
        self.student = MultiCropWrapper(
                    backbone = student_backbone,
                    head = MLPHead(
                        input_dim = student_backbone.embed_dim, 
                        output_dim = head_output_dim, 
                        hidden_dim = head_hidden_dim,
                        bottleneck_dim = None,
                        use_bn = head_use_bn, 
                        norm_last_layer = head_norm_last_layer, 
                        nlayers = head_nlayers,
                        init = "trunc_normal",
                        norm_layer = head_norm_layer,
                        activation = head_activation,
                    ),
                    use_dense_prediction=False,
                    head_dense=None,
                    num_prototypes = num_prototypes,
                )
        self.student = self.student.to(device)
        self.logger.info("Built backbone network.")
        self.num_crops = num_crops
        self.crops_for_assign = crops_for_assign 
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.num_prototypes = num_prototypes
        self.queue_length = queue_length
        self.epoch_queue_starts = epoch_queue_starts
        self.freeze_prototypes_niters = freeze_prototypes_niters
        self.clip_grad_magnitude = clip_grad_magnitude
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
        # criterion
        self.criterion = SWAVLoss(
                        crops_for_assign,
                        num_crops,
                        batch_size,
                        temperature,
                        head_output_dim,
                        queue_length,
                        epoch_queue_starts,
                        sinkhorn_iterations,
                        epsilon = epsilon,
                        device = self.device,
                    ).to(self.device)
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
        # optimizer
        # lr is assigned by scheduler
        try:
            if optimizer == "SGD":
                optim = torch.optim.SGD(
                    self.student.parameters(),
                    lr=0.,
                    momentum=0.9,
                )
                self.optimizer = LARC(optimizer=optim, trust_coefficient=0.001, clip=False)
            else:
                raise NotImplementedError(f"{optimizer} not implemented")
        except NotImplementedError as error:
            self.logger.error(error)
            raise error

        self.logger.info(f"Initialized {optimizer} Optimizer")
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
        # scheduler
        self.lr_schedule = cosine_scheduler(
            init_val = lr, 
            final_val = final_lr,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
            start_warmup_value = start_warmup_lr,
            warmup_epochs = scheduler_warmup_epochs,
        )
        self.wd_schedule = cosine_scheduler(
            init_val = weight_decay,
            final_val = final_weight_decay,
            epochs = self.epochs, 
            start_warmup_value = weight_decay,
            steps_per_epoch = self.steps_per_epoch,
        )
        torch.backends.cudnn.benchmark = True
        self.logger.info(f"Initialized schedulers")
    #--------------------------------------------------------------------------------------------------------
        self._init_state()
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
    def forward_teacher(self, img):
        return None
    #--------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------
    def forward_student(self, img, *args, **kwargs):
    #--------------------------------------------------------------------------------------------------------
        # normalize the prototypes
        with torch.no_grad():
            w = self.student.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.student.prototypes.weight.copy_(w)
    #--------------------------------------------------------------------------------------------------------
        return self.student(img)
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
    def set_lr_and_wd(self, current_step):
        lr, wd = self.lr_schedule[current_step], self.wd_schedule[current_step]
        for param_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
            if param_idx == 0:  
                param_group["weight_decay"] = wd
    #--------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------            
    def compute_loss_epoch(self, student_output, teacher_output, mask):
        return self.criterion(student_output, epoch=self.state["current_epoch"], prototypes_weight = self.student.prototypes.weight)
    #--------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------
    def process_grads_before_step(self, epoch, iter): 
        # cancel gradients for the prototypes
        if iter < self.freeze_prototypes_niters:
            for name, p in self.student.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        if self.clip_grad_magnitude > 0:
            if self.fp16_scaler is not None:
                self.fp16_scaler.unscale_(self.optimizer)
            for param in self.student.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    clip_coef = self.clip_grad_magnitude / (param_norm + 1e-6)
                    if clip_coef < 1:
                        param.grad.data.mul_(clip_coef)
    #--------------------------------------------------------------------------------------------------------
      

    #--------------------------------------------------------------------------------------------------------
    def _cancel_gradients_last_layer(self, epoch, model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None
    #--------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------
    def update_teacher(self, current_step):
        return None
    #--------------------------------------------------------------------------------------------------------

        






        
