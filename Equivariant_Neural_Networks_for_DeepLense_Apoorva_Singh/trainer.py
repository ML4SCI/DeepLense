import sys
import os
import time
import logging
import argparse
import shutil
import torch
import torchvision
import numpy as np
from os import listdir
import pandas as pd
from e2cnn import gspaces
from e2cnn import nn
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import join
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
import torchvision.transforms as transforms
from model import Equivariant_Network
from resnet_model import resnet18
from sklearn.metrics import roc_curve, auc, confusion_matrix





class ECNN(object):

    def __init__(self,logger, log_dir, current_time, n_classes=3, sym_group = "Dihyderal", N = 4,device = 'cpu',lr=5e-5, use_CNN = False, mode = 'Train'):

        self.lr = lr
        self.N = N
        self.n_classes = n_classes
        self.sym_group = sym_group
        self.use_CNN = use_CNN
        #DEFINE NETWORK
        if self.use_CNN:
            self.model = resnet18(pretrained = False, progress = False, num_classes = self.n_classes).to(device)
            
        else:
            self.model = Equivariant_Network(n_classes=self.n_classes, sym_group = self.sym_group, N = self.N).to(device)
        if mode == 'Train':
            #DEFINE LOSS
            self.loss_function = torch.nn.CrossEntropyLoss()

            #OPTIMIZER
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.device = device
        
        logger.setLevel(logging.DEBUG)
        
        self.log_dir = log_dir
        self.current_time = current_time

        if mode == 'Test':
            self.model.load_state_dict(torch.load('{}/best-model-parameters.pt'.format(os.path.join(self.log_dir,self.current_time),map_location=self.device)))



    def train(self,train_loader,test_loader,total_epochs):
        log_interval = 100
        self.all_train_loss = []
        self.all_test_loss = []
        self.all_train_accuracy = []
        self.all_test_accuracy = []
        self.total_epochs = total_epochs

#         self.log_dir = log_dir
#         self.current_time = current_time

        best_accuracy = 0

        for epoch in range(self.total_epochs):
            self.model.train()
            tr_loss_epoch = []
            test_loss_epoch = []
            total = 0
            correct = 0
            for i, (x, t) in enumerate(train_loader):        
                self.optimizer.zero_grad()

                x = x.to(self.device)
                t = t.to(self.device)

                y = self.model(x)
                y_pred = y.flatten().to(torch.float64)
                
                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                
        
                loss = self.loss_function(y, t)
                tr_loss_epoch.append(loss.item())
                if i % log_interval == 0:
                    logging.debug('Loss: {}'.format(loss.item()))
                loss.backward()

                self.optimizer.step()
              
            self.all_train_loss.append(np.asarray(tr_loss_epoch))
            self.all_train_accuracy.append(correct/total*100)
            
            
            if epoch % 1 == 0:
                total = 0
                correct = 0
                with torch.no_grad():
                    self.model.eval()
                    for i, (x, t) in enumerate(test_loader):
                        x = x.to(self.device)
                        t = t.to(self.device)
                        y = self.model(x)
                    
                        loss = self.loss_function(y, t)
                        test_loss_epoch.append(loss.item())

                        _, prediction = torch.max(y.data, 1)
                        total += t.shape[0]
                        correct += (prediction == t).sum().item()
                        
                self.all_test_loss.append(np.asarray(test_loss_epoch))
                self.all_test_accuracy.append(correct/total*100)
                logging.debug("epoch {} | test accuracy: {}".format(epoch,correct/total*100))
                
                test_accuracy = correct/total*100
                
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    torch.save(self.model.state_dict(), '{}/best-model-parameters.pt'.format(os.path.join(self.log_dir, self.current_time)))

        return self.all_train_loss,self.all_test_loss,self.all_train_accuracy,self.all_test_accuracy


    def save_results(self):

        all_epochs = [i for i in range (self.total_epochs)]

        all_train_loss_mean = [j.mean() for j in self.all_train_loss]
        all_test_loss_mean = [j.mean() for j in self.all_test_loss]

        np.save("{}/train_loss.npy".format(os.path.join(self.log_dir, self.current_time)), all_train_loss_mean)
        np.save("{}/test_loss.npy".format(os.path.join(self.log_dir, self.current_time)), all_test_loss_mean)
        np.save("{}/train_accuracy.npy".format(os.path.join(self.log_dir, self.current_time)), self.all_train_accuracy)
        np.save("{}/test_accuracy.npy".format(os.path.join(self.log_dir, self.current_time)), self.all_test_accuracy)

        params_mat = {'legend.fontsize': 5.5,
                  'axes.labelsize': 6,
                  'axes.titlesize': 5,
                  'xtick.labelsize': 5,
                  'ytick.labelsize': 5,
                  'axes.titlepad': 25}
        plt.rcParams.update(params_mat)

        fig = plt.figure(figsize=(6, 3), dpi=600)


        plt.grid(color='black', linestyle='dotted', linewidth=.5, alpha = 0.5)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.plot(all_epochs, all_train_loss_mean, label="Training Loss")
        plt.plot(all_epochs, all_test_loss_mean, label="Testing Loss")

        plt.savefig("{}/loss.png".format(os.path.join(self.log_dir, self.current_time)), dpi = 600)

        plt.legend()


        params_mat = {'legend.fontsize': 5.5,
                  'axes.labelsize': 6,
                  'axes.titlesize': 5,
                  'xtick.labelsize': 5,
                  'ytick.labelsize': 5,
                  'axes.titlepad': 25}
        plt.rcParams.update(params_mat)

        fig = plt.figure(figsize=(6, 3), dpi=600)

        plt.grid(color='black', linestyle='dotted', linewidth=.5, alpha = 0.5)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.plot(all_epochs, self.all_train_accuracy, label="Training Accuracy")
        plt.plot(all_epochs, self.all_test_accuracy, label="Testing Accuracy")

        plt.savefig("{}/accuracy.png".format(os.path.join(self.log_dir, self.current_time)), dpi = 600)

        plt.legend()
    
    def __to_one_hot_vector(self,num_class, label):
        b = np.zeros((label.shape[0], num_class))
        b[np.arange(label.shape[0]), label] = 1

        return b.astype(int)
    
    
    def plot_ROC(self, testset, test_loader):
        
        plt.rcParams.update(plt.rcParamsDefault)

        total = 0
        all_test_loss = []
        all_test_accuracy = []
        label_true_arr = []
        label_pred_arr = []

        correct = 0
        with torch.no_grad():
            self.model.eval()
            for i, (x, t) in enumerate(test_loader):
                x = x.to(self.device)
                t = t.to(self.device)
                y = self.model(x)

                label_pred_arr.append(y.cpu().numpy())


                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                print(correct/total*100)

                one_hot_t = self.__to_one_hot_vector(3,t.cpu().numpy())
                label_true_arr.append(one_hot_t)

        y_test = []
        for i in label_true_arr:
            for j in i:
                y_test.append(list(j))
        y_test = np.array(y_test)

        y_score = []
        for i in label_pred_arr:
            for j in i:
                y_score.append(list(j))
        y_score = np.array(y_score)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
#         n_classes = 3
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        inv_map = {v: k for k, v in testset.class_to_idx.items()}



        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.4f})'
                       ''.format(roc_auc["micro"]))
        for i in range(self.n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class '+ inv_map[i]+ ' (area = {0:0.4f})'
                                           ''.format(roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if self.use_CNN == False:
            plt.title(self.sym_group[0]+str(self.N)+' ROC')
        else:
            plt.title('ResNet-18 ROC')
        plt.legend(loc="lower right")
        plt.savefig("{}/ROC.png".format(os.path.join(self.log_dir, self.current_time)), dpi = 600)

        
        
    def __plot_confusion_matrix(self, cm, classes,save_path,
                                      normalize=False,
                                      title='Confusion matrix',
                                      cmap=plt.cm.Blues):
        
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.rcParams.update(plt.rcParamsDefault)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(save_path,dpi=400)    
        

    def plot_CM(self, testset, test_loader):
        plt.rcParams.update(plt.rcParamsDefault)
        total = 0
        label_pred_arr = []
        label_true_arr = []


        correct = 0
        with torch.no_grad():
            self.model.eval()
            for i, (x, t) in enumerate(test_loader):
                x = x.to(self.device)
                t = t.to(self.device)
                y = self.model(x)

                _, prediction = torch.max(y.data, 1)
                label_pred_arr.append(prediction.cpu().numpy())
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                print(correct/total*100)
                label_true_arr.append(t.cpu().numpy())



        y_pred = []
        for i in label_pred_arr:
            for j in i:
                y_pred.append(j)
        y_pred = np.array(y_pred)

        y_true = []
        for i in label_true_arr:
            for j in i:
                y_true.append(j)
        y_true = np.array(y_true)


                         


        inv_map = {v: k for k, v in testset.class_to_idx.items()}

        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        if self.use_CNN == False:
            self.__plot_confusion_matrix(cnf_matrix, classes=[inv_map[0], inv_map[1], inv_map[2]],save_path = "{}/CM.png".format(os.path.join(self.log_dir, self.current_time)),title='Confusion matrix for '+self.sym_group[0]+str(self.N))
        else:
            self.__plot_confusion_matrix(cnf_matrix, classes=[inv_map[0], inv_map[1], inv_map[2]],save_path = "{}/CM.png".format(os.path.join(self.log_dir, self.current_time)),title='Confusion matrix for ResNet-18')
            









