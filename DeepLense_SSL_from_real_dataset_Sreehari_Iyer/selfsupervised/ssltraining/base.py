import os
import time
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import random_split, DataLoader, Dataset
from selfsupervised.utils import knn_accuracy
import datetime
from typing import List, Dict, Union, Callable 
from selfsupervised.augmentations import ImageDataset, ImageDatasetMasked

def npy_loader(path):
    sample = torch.from_numpy(np.load(path).astype(np.float32))
    return sample

class TrainSSL:
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            restore_from_ckpt: bool,
            restore_ckpt_path: str,
            data_path,
            train_test_indices,
            data_augmentation_transforms,
            eval_transforms,
            num_classes: int,
            train_val_split: List[float, ],
            batch_size: int,
            num_epochs: int,
            log_freq: int,
            device: str,
            logger,
            patch_size=None,
            pred_ratio=None,
            pred_ratio_var=None,
            pred_aspect_ratio=None,
            pred_shape=None,
            pred_start_epoch=None,
            use_mixed_precision: bool=True,
            knn_neighbours: int = 20,
            use_dense_prediction: bool = False,
            masked_loader: bool = False,
            **kwargs,
        ):

        self.masked_loader = masked_loader
        self.output_dir = output_dir
        self.expt_name = expt_name
        self.expt_path = os.path.join(output_dir, f"{expt_name}_models")
        if not os.path.exists(self.expt_path):
            os.makedirs(self.expt_path)
        self.ckpt_file = os.path.join(self.expt_path, "checkpoint.pth")
        self.restore_ckpt_path = None
        if restore_from_ckpt:
            self.restore_ckpt_path= restore_ckpt_path
        self.log_freq = log_freq
        self.logger = logger 
        self.epochs = num_epochs
        self.optimizer = None
        self.criterion = None
        self.device = device   
        self.use_mixed_precision = use_mixed_precision
        self.student = None
        self.teacher = None
        self.num_classes = num_classes
        self.data_path = data_path
        self.batch_size = batch_size
        self.knn_neighbours = knn_neighbours
        self.use_dense_prediction = use_dense_prediction
        assert data_augmentation_transforms is not None
        assert eval_transforms is not None

        indices = None
        with open(train_test_indices, "rb") as f:
            indices = pickle.load(f)
        train_indices = indices["train"]





            # train images and labels
        train_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(train_indices["lenses"])] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(train_indices["nonlenses"])]
        train_labels = [0]*len(train_indices["lenses"]) + [1]*len(train_indices["nonlenses"])

        # validation images and labels
        val_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["val"]["lenses"])] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["val"]["nonlenses"])]
        val_labels = [0]*len(indices["val"]["lenses"]) + [1]*len(indices["val"]["nonlenses"])

        # test images and labels
        test_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["test"]["lenses"])] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["test"]["nonlenses"])]
        test_labels = [0]*len(indices["test"]["lenses"]) + [1]*len(indices["test"]["nonlenses"])

        # sizes of the train, validation and test datasets
        train_size = len(train_paths)
        val_size = len(val_paths)
        test_size = len(test_paths)
        #
        # Create DatasetFolder instances for each split with the respective transforms

        # training dataset 
        dataset = ImageDataset(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=data_augmentation_transforms,
            return_indices=True,
        ) if not masked_loader else \
        ImageDatasetMasked(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=data_augmentation_transforms,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch
        )

        # train split of the dataset for evaluation
        train_dataset = ImageDataset(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=eval_transforms,
            return_indices=True,
        ) if not masked_loader else \
        ImageDatasetMasked(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=eval_transforms,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch
        )
        # validation split of the dataset for evaluation
        val_dataset = ImageDataset(
            image_paths=val_paths,
            labels=val_labels,
            loader=npy_loader,
            transform=eval_transforms,
            return_indices=True,
        ) if not masked_loader else \
        ImageDatasetMasked(
            image_paths=val_paths,
            labels=val_labels,
            loader=npy_loader,
            transform=eval_transforms,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch
        )
        # test split of the dataset for evaluation
        test_dataset = ImageDataset(
            image_paths=test_paths,
            labels=test_labels,
            loader=npy_loader,
            transform=eval_transforms,
            return_indices=True,
        ) if not masked_loader else \
        ImageDatasetMasked(
            image_paths=test_paths,
            labels=test_labels,
            loader=npy_loader,
            transform=eval_transforms,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_aspect_ratio=pred_aspect_ratio,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch
        )

        # Dataloaders
        
        # train
        self.dataloader = DataLoader(
            dataset,
            # batch_size = self.physical_batch_size,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', True),
        )

        # evaluation
        self.eval_train_dataloader = DataLoader(
            dataset=train_dataset,
            # batch_size = self.physical_batch_size,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', False),
        )
        self.eval_val_dataloader = None
        if val_size!=0:
            self.eval_val_dataloader = DataLoader(
                dataset=val_dataset,
                # batch_size = self.physical_batch_size,
                batch_size = self.batch_size,
                num_workers = kwargs.get('num_workers', 4),
                pin_memory = kwargs.get('pin_memory', True),
                drop_last = kwargs.get('drop_last', False),
            ) 
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            # batch_size = self.physical_batch_size,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', False),
        )    
        self.logger.info(f"Loaded Dataset from {self.data_path}")
        self.logger.info(f"Train Dataset contains {train_size} images")
        self.logger.info(f"Val Dataset contains {val_size} images")
        self.logger.info(f"Test Dataset contains {test_size} images")
        self.steps_per_epoch = len(self.dataloader)

        # initialize history dict
        self.history = {
            "loss_stepwise": [],
            "loss_epochwise": [],
            "knn_top1": [],
            "knn_top5": [],
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            # "lens_train_indices": lens_train_indices,
            # "nonlens_train_indices": nonlens_train_indices,
            # "lens_val_indices": lens_val_indices,
            # "nonlens_val_indices": nonlens_val_indices,
        }
        self.state = None

    def _init_state(self):
        # initialize the state variable
        self.state = {
            "info": {
                "output_dir": self.output_dir,
                "expt_name": self.expt_name,
                "ckpt_file": self.ckpt_file,
                "self.device": self.device,
            }, 
            "student": self.student,
            "teacher": self.teacher,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "history": self.history,
            "current_epoch": 0,
            "lr": self.lr_schedule, 
            "wd": self.wd_schedule
        }

        self.fp16_scaler = None
        if self.use_mixed_precision:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            self.state["fp16_scaler"]: self.fp16_scaler

        if self.restore_ckpt_path is not None:
            self._restore()

    
    def _restore(self) -> None:
        assert self.restore_ckpt_path is not None, "Checkpoint path to restore not provided"
        
        ckpt_dict = torch.load(self.restore_ckpt_path, map_location="cpu")

        for restore_item in self.state:
            if restore_item == "info":
                continue
            elif restore_item in ["student", "teacher", "fp16_scaler"]:
                msg = self.state[restore_item].load_state_dict(ckpt_dict[restore_item].state_dict(), strict=False)
            else:
                msg = self.state[restore_item].load_state_dict(ckpt_dict[restore_item], strict=False)
            self.logger.info(f"Loaded {restore_item} from checkpoint {self.restore_ckpt_name}\n\t{msg}")

        self.logger.info(f"Restored checkpoint from {self.restore_ckpt_name}")
                
    def update_history(self, step: Dict)->None:
        for key in self.history:
            if step[key] is not None:
                self.history[key].append(step[key])

    def save_checkpoint(self, ckpt_file):
        torch.save(self.state, ckpt_file)

    def forward_teacher(self, img):
        return self.teacher(img) 

    def forward_student(self, img):
        return self.student(img)

    def update_teacher(self):
        raise NotImplementedError

    def process_grads_before_step(self, epoch): 
        return None

    def set_lr_and_wd(self, current_step):
        raise NotImplementedError

    @torch.no_grad()
    def update_teacher(self):
        raise NotImplementedError

    # train for one epoch
    def train_one_epoch(self):

        losses = []
        cur_step = self.state["current_epoch"]*self.steps_per_epoch 
        for idx, input in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            img, mask = None, None 
            imgs, label, msk = None, None, None 
            indices = None
            if self.masked_loader:
                imgs, label, msk = input
                img = [im.to(self.device, non_blocking=True) for im in imgs]
                mask = [im.to(self.device, non_blocking=True) for im in msk]
            else:
                imgs, label, indices = input
                img = [im.to(self.device, non_blocking=True) for im in imgs]

            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                teacher_output = self.forward_teacher(img) if self.teacher is not None else None
                _img = (img, mask) if mask is not None else img
                student_output = self.forward_student(_img, labels=label, indices=indices)
                loss = self.compute_loss_epoch(student_output=student_output, teacher_output=teacher_output, mask=mask)
                losses.append(loss.item())
    
            if np.isinf(loss.item()):
                self.logger.error(f"Loss is Infinite. Training stopped")
                sys.exit(1)

            self.set_lr_and_wd(cur_step+idx)
            
            if self.fp16_scaler is None:
                loss.backward()
                self.process_grads_before_step(self.state["current_epoch"], cur_step+idx)
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                self.process_grads_before_step(self.state["current_epoch"], cur_step+idx)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            self.optimizer.zero_grad()

            if self.teacher is not None:
                self.update_teacher(cur_step)

        self.history["loss_stepwise"].extend(losses)
        self.history["loss_epochwise"].append(np.mean(losses))
        return np.mean(losses)


    def train(self):
        if self.state is None:
            self._init_state()
        for epoch in range(self.state["current_epoch"], self.epochs): # this will help to restart training
            self.student.train() 
            if self.teacher is not None:
                self.teacher.train()
            start_time = time.time()
            loss = self.train_one_epoch()
            total_time = time.time() - start_time
            self.logger.info(f"Epoch: {epoch} finished in {datetime.timedelta(seconds=int(total_time))} seconds,")
            self.logger.info(f"\tTrain Loss: {loss:.6e}")
            
            knn_top1 = None
            self.student.eval()
            knn_top1 = self.compute_knn_accuracy()

            self.history["knn_top1"].append(knn_top1)

            self.save_checkpoint(self.ckpt_file)
            if self.state["current_epoch"]%self.log_freq == 0 or (self.state["current_epoch"] + 1) == self.epochs:
                self.save_checkpoint(os.path.join(self.expt_path, f"epoch_{self.state['current_epoch']}_accknn_{knn_top1:.6f}_checkpoint.pth"))
            self.state["current_epoch"] += 1
        

        self.student.eval() 
        knn_top1 = self.compute_knn_accuracy(mode = "Test")
        self.history["Test"] = {
            "knn_top1": knn_top1,
        }

        self.save_checkpoint(self.ckpt_file)

    @torch.no_grad()
    def compute_knn_accuracy(
            self, 
            mode: str = None,
            knn_k: int= None,
        ):
        self.logger.info("Testing accuracy of learned features using KNN")
        test_dataloader_ = None
        if mode is None:
            if self.eval_val_dataloader is not None:
                mode = "Val"
            else:
                mode = "Test"
        if knn_k is None:
            knn_k = self.knn_neighbours
        if mode == "Val":
            test_dataloader_=self.eval_val_dataloader
        elif mode == "Test":
            test_dataloader_=self.test_dataloader
        else:
            self.logger.error(f"mode {mode} not defined")
        top1, _ = knn_accuracy(
            model=self.student.backbone,
            train_dataloader=self.eval_train_dataloader,
            test_dataloader=test_dataloader_,
            classes=self.num_classes, 
            knn_k=knn_k,
            device=self.device,
            use_dense_prediction=self.student.use_dense_prediction,
            masked_loader = self.masked_loader,
            return_all_tokens = self.student.backbone.return_all_tokens,
        )
        self.logger.info(f"\t{mode}: KNN Acc@1:{top1:.6f} with neighbour count: {knn_k}")
        return top1
    
        
        

        
