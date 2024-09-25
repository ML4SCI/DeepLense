import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import TrainSSL
from selfsupervised.models import MLPHead, MultiCropWrapper
from selfsupervised.utils import cosine_scheduler
from typing import List, Dict, Union, Optional, Tuple

class TrainSIMSIAM(TrainSSL):
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            logger,
            student_backbone: nn.Module,
            data_path,
            train_test_indices,
            data_augmentation_transforms,
            eval_transforms,
            num_classes: int,
            train_val_split: Tuple[float, ],
            batch_size: int,

            embed_dim: int,
            projector_head_hidden_dim: int,
            head_output_dim: int,
            projector_use_bn: bool, 
            projector_nlayers: int,

            predictor_head_hidden_dim: int,

            lr: float = 1e-3,
            final_lr: Optional[float] = None,
            weight_decay: float = 5e-3,
            final_weight_decay: Optional[float] = None,
            scheduler_warmup_epochs: int = 10,
            num_epochs: int = 100,
            fix_pred_lr: bool = True,
            clip_grad_magnitude: float = 0.,
            restore_from_ckpt: bool = False,
            restore_ckpt_path: Optional[str] = None,

            optimizer: Union["AdamW, Adam, SGD, LARS"] = "SGD",
            log_freq: int = 5,
            device: str = "cuda",
            use_mixed_precision: bool = "True",
            freeze_last_layer: int = 1,
            knn_neighbours: int = 5,
            **kwargs,
        ):
        
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
            use_dense_prediction = False,
            **kwargs
        )
        self.logger.info("Initializing DINO training")

        self.clip_grad_magnitude = clip_grad_magnitude
        self.freeze_last_layer = freeze_last_layer
        
        # student and teacher network
        self.student = MultiCropWrapper(
                    backbone = student_backbone,
                    head = MLPHead(
                        input_dim = embed_dim, 
                        output_dim = head_output_dim, 
                        hidden_dim = projector_head_hidden_dim,
                        bottleneck_dim = None,
                        use_bn = projector_use_bn, 
                        norm_last_layer = True, 
                        nlayers = projector_nlayers,
                        normalize_outputs = False,
                        activation = nn.ReLU,
                        norm_layer = nn.BatchNorm1d,
                    ),
                    use_dense_prediction=self.use_dense_prediction,
                    head_dense = None,
                )
        self.predictor = MLPHead(
                            input_dim = head_output_dim, 
                            output_dim = head_output_dim, 
                            hidden_dim = predictor_head_hidden_dim,
                            bottleneck_dim = None,
                            use_bn = True, 
                            norm_last_layer = False, 
                            nlayers = 1,
                            normalize_outputs = False,
                            activation = nn.ReLU,
                            norm_layer = nn.BatchNorm1d,
                        )
        self.teacher = None
        self.student = self.student.to(device)
        self.predictor = self.predictor.to(device)
        
        self.logger.info("Built Student and Teacher networks.\nBoth are initialized with same parameters")

        # loss
    
        self.criterion = nn.CosineSimilarity(dim=1).to(self.device)
        
        # optimizer
        # lr is assigned by scheduler
        optim_params = []
        for param in self.student.parameters():
            optim_params.append(param)
        for param in self.predictor.parameters():
            optim_params.append(param)
        try:
            if optimizer == "AdamW":
                self.optimizer = torch.optim.AdamW(optim_params)
            elif optimizer == "Adam":
                self.optimizer = torch.optim.Adam(optim_params) 
            elif optimizer == "SGD":
                self.optimizer = torch.optim.SGD(optim_params, lr=0., momentum=0.9) 
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
        self.logger.info(f"Initialized schedulers")

        self._init_state()

    def forward_teacher(self, img):
        return None

    def forward_student(self, img, *args, **kwargs):
        e0 = self.student(x=img[0], return_backbone_feat=False)
        e1 = self.student(x=img[1], return_backbone_feat=False)
        p0, p1 = self.predictor(e0).detach(), self.predictor(e1).detach()
        return (e0, e1), (p0, p1)

    def set_lr_and_wd(self, current_step):
        lr, wd = self.lr_schedule[current_step], self.wd_schedule[current_step]
        for param_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
            if param_idx == 0:  
                param_group["weight_decay"] = wd

    def compute_loss_epoch(self, student_output, teacher_output, mask):
        encoder_outputs, predictor_outputs = student_output
        e0, e1 = encoder_outputs
        p0, p1 = predictor_outputs
        # print(len(p1), p1[0].shape)
        return -(self.criterion(e0, p1).mean() + self.criterion(e1, p0).mean()) * 0.5


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


        






        
