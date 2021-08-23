import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns

class Supervised(object):
    def __init__(self, encoder, classifier):
        """
        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)

    def train(self, dataloader, dataloader_test, epochs, hyperparams, save_path):
        """
        Trains the model (encoder + classifier).

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with training data.

        dataloader_test: PyTorch DataLoader
            DataLoader with validation data, used for early stopping.

        epochs: int
            Amount of epochs to train the model for.

        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.

        save_path: str
            Path to store model weights.

        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """
        
        # configure hyperparameters
        criterion = nn.CrossEntropyLoss()
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        cyclic_scheduler = hyperparams['cyclic_scheduler']

        # concatenate parameters of both self.encoder and self.classifier
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=wd)
        if cyclic_scheduler:
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(dataloader))

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 15
        bad_epochs = 0

        # metrics
        self.history = {'loss': [], 'accuracy': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            # set network to training mode
            self.encoder.train()
            self.classifier.train()

            running_loss = 0.0

            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # zero gradients
                optimizer.zero_grad()

                # classify and get loss
                outputs = self.classifier(self.encoder(data)) # Cs(Ms(xs))
                loss = criterion(outputs, labels)

                # backpropagate and update weights
                loss.backward()
                optimizer.step()

                if cyclic_scheduler:
                    scheduler.step()

                # metrics
                running_loss += loss.item()

            # get losses
            epoch_loss = running_loss / len(dataloader)
            self.history['loss'].append(epoch_loss)

            # evaluate on training data (source domain)
            epoch_accuracy = self.evaluate( dataloader)
            test_epoch_accuracy = self.evaluate(dataloader_test)
            self.history['accuracy'].append(epoch_accuracy)
            
            # save checkpoint
            if test_epoch_accuracy > best_acc:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                            'classifier_weights': self.classifier.state_dict()
                        }, save_path)
                best_acc = test_epoch_accuracy
                bad_epochs = 0
                
            else:
                bad_epochs += 1
            
            print('[Epoch {}/{}] loss: {:.6f}; accuracy: {:.6f}; val accuracy: {:.6f};'.format(epoch+1, epochs, epoch_loss, epoch_accuracy, test_epoch_accuracy))
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])
        
        return self.encoder, self.classifier

    def evaluate(self, dataloader, return_lists_roc=False):
        """
        Evaluates model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.

        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.encoder.eval()
        self.classifier.eval()

        labels_list = []
        outputs_list = []
        preds_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # predict
                outputs = F.softmax(self.classifier(self.encoder(data)), dim=1)

                # numpify
                labels_numpy = labels.detach().cpu().numpy()
                outputs_numpy = outputs.detach().cpu().numpy() # probs (AUROC)
                
                preds = np.argmax(outputs_numpy, axis=1) # accuracy

                # append
                labels_list.append(labels_numpy)
                outputs_list.append(outputs_numpy)
                preds_list.append(preds)

            labels_list = np.concatenate(labels_list)
            outputs_list = np.concatenate(outputs_list)
            preds_list = np.concatenate(preds_list)

        # metrics
        #auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list
            
        return accuracy

    def plot_metrics(self):
        """
        Plots the training metrics (only usable after calling .train()).
        """

        # plot metrics from source
        fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=200)

        epochs = len(self.history['loss'])

        axs[0].plot(range(1, epochs+1), self.history['loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Source loss')

        axs[1].plot(range(1, epochs+1), self.history['accuracy'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Source accuracy')

        plt.show()

    def plot_cm_roc(self, dataloader, n_classes=3):
        """
        Plots the confusion matrix and ROC curves of the model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        n_classes: int
            Number of classes.
        """

        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        self.encoder.eval()
        self.classifier.eval()

        accuracy, labels_list, outputs_list, preds_list = self.evaluate(dataloader, return_lists_roc=True)

        # plot confusion matrix
        cm = sklearn.metrics.confusion_matrix(labels_list, preds_list)
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(3,3)
        #tn, fp, fn, tp = cm.ravel()

        plt.figure(figsize=(4,4), dpi=200)
        categories = ['no', 'spherical', 'vortex'] #  0:no substructure 1:spherical substructure 2:vortex substructure  
        sns.heatmap(cm, annot=labels, cmap=cmap, fmt="", xticklabels=categories, yticklabels=categories)
        plt.title("Confusion matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.show()

        # plot roc
        ## one hot encode data
        onehot = np.zeros((labels_list.size, labels_list.max()+1))
        onehot[np.arange(labels_list.size),labels_list] = 1
        onehot = onehot.astype('int')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        ## get roc curve and auroc for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(onehot[:, i], outputs_list[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        ## get macro average auroc
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(6,6), dpi=200)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        c = ['blue', 'green', 'orange']
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label=f"AUC class {i} = {roc_auc[i]:.4f}", color=c[i])

        plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average AUC = {roc_auc['macro']:.4f}", color='deeppink', linewidth=2)
            
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.show()