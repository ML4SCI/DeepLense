from .supervised import Supervised
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.spatial.distance import cdist # cosine distance
from torch.autograd import grad

class Cgdm(Supervised):
    """
    Paper: Cross-Domain Gradient Discrepancy Minimization for Unsupervised Domain Adaptation
    Authors: Zhekai Du, Jingjing Li, Hongzu Su, Lei Zhu, Ke Lu
    """

    def __init__(self, encoder, classifier1, classifier2):
        """
        Here, `classifier1` and `classifier2` are the bi-classifiers.
        They can have different parameters, but must have the same architecture.

        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier1: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
            If using transfer learning from the source, this should be the source classifier.

        classifier2: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
            If using transfer learning from the source, this should ALSO be the source classifier.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)

        self.classifier1 = classifier1.to(self.device)
        self.classifier2 = classifier2.to(self.device)

    def train(self, source_dataloader, target_dataloader, target_dataloader_test, epochs, hyperparams, save_path):
        """
        Trains the model (encoder + classifier1 + classifier2).

        Arguments:
        ----------
        source_dataloader: PyTorch DataLoader
            DataLoader with source domain training data.

        target_dataloader: PyTorch DataLoader
            DataLoader with target domain training data.

        target_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.

        epochs: int
            Amount of epochs to train the model for.

        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.

        n_classes: int
            Number of classes.

        save_path: str
            Path to store model weights.

        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier1: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.

        classifier2: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        # configure hyperparameters
        criterion = nn.CrossEntropyLoss()
        criterion_weighted = self._weighted_crossentropy

        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        num_k = hyperparams['num_k']
        cyclic_scheduler = hyperparams['cyclic_scheduler']
        pseudo_interval = hyperparams['pseudo_interval']
        
        iters = max(len(source_dataloader), len(target_dataloader))

        # configure optimizer and scheduler
        optimizer_encoder = optim.Adam(list(self.encoder.parameters()), lr=lr, weight_decay=wd)
        optimizer_classifiers = optim.Adam(list(self.classifier1.parameters()) + list(self.classifier2.parameters()), lr=lr, weight_decay=wd)

        if cyclic_scheduler:
            scheduler_encoder = optim.lr_scheduler.OneCycleLR(optimizer_encoder, lr, epochs=epochs, steps_per_epoch=iters)
            scheduler_classifiers = optim.lr_scheduler.OneCycleLR(optimizer_classifiers, lr, epochs=epochs, steps_per_epoch=iters)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 15
        bad_epochs = 0

        self.history = {'epoch_loss_entropy': [], 'epoch_loss_classifier1': [], 'epoch_loss_classifier2': [], 'epoch_loss_discrepancy': [], 'accuracy_source': [], 'accuracy_target': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss_entropy = 0.0
            running_loss_classifier1 = 0.0
            running_loss_classifier2 = 0.0
            running_loss_discrepancy = 0.0

            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for i, ((data_source, labels_source), (data_target, _)) in enumerate(zip(source_dataloader, target_dataloader)):
                if epoch > start_epoch and i % pseudo_interval == 0:
                    pseudo_labels_target = self._get_pseudo_labels(target_dataloader)

                # set network to training mode
                self.encoder.train()
                self.classifier1.train()
                self.classifier2.train()

                data_source = data_source.to(self.device)
                labels_source = labels_source.to(self.device)

                data_target = data_target.to(self.device)
                if epoch > start_epoch:
                    labels_target = pseudo_labels_target[target_dataloader.batch_size*i : target_dataloader.batch_size*(i+1)]
                    labels_target = labels_target.to(self.device)

                # all steps have similar begginings
                for phase in [1, 2, 3]:
                    for k in range(num_k): # amount of steps to repeat the self.encoder update
                        # zero gradients
                        optimizer_encoder.zero_grad()
                        optimizer_classifiers.zero_grad()
                        
                        # classify the data
                        features_source = self.encoder(data_source)
                        features_target = self.encoder(data_target)

                        outputs1_source = self.classifier1(features_source)
                        outputs1_target = self.classifier1(features_target)
                        outputs2_source = self.classifier2(features_source)
                        outputs2_target = self.classifier2(features_target)

                        # get losses
                        entropy_loss = self._entropy(outputs1_target) + self._entropy(outputs2_target)

                        loss1 = criterion(outputs1_source, labels_source)
                        loss2 = criterion(outputs2_source, labels_source)

                        if phase == 1:
                            # train networks to minimize loss on source
                            if epoch > start_epoch:
                                supervised_loss = criterion_weighted(outputs1_target, labels_target) + criterion_weighted(outputs2_target, labels_target)

                            else:
                                supervised_loss = 0

                            loss = loss1 + loss2 + (0.01 * entropy_loss) + (0.01 * supervised_loss)

                            # backpropagate and update weights
                            loss.backward()
                            optimizer_encoder.step()
                            optimizer_classifiers.step()

                            # exit self.encoder loop (num_k)
                            break

                        elif phase == 2:
                            # train classifiers to maximize divergence between classifier outputs on target (without labels)
                            discrepancy_loss = self._discrepancy(outputs1_target, outputs2_target)
                            loss = loss1 + loss2 - (1.0 * discrepancy_loss) + (0.01 * entropy_loss) 
                            
                            # backpropagate and update weights
                            loss.backward()
                            optimizer_classifiers.step()

                            # exit self.encoder loop (num_k)
                            break

                        elif phase == 3:
                            # train self.encoder to minimize divergence between classifier outputs with gradient similarity
                            discrepancy_loss = self._discrepancy(outputs1_target, outputs2_target)

                            if epoch > start_epoch:
                                source_pack = (outputs1_source, outputs2_source, labels_source)
                                target_pack = (outputs1_target, outputs2_target, labels_target)
                                gradient_discrepancy_loss = self._gradient_discrepancy(source_pack, target_pack)
                            else:
                                gradient_discrepancy_loss = 0

                            loss = (1.0 * discrepancy_loss) + (0.01 * entropy_loss) + (0.01 * gradient_discrepancy_loss)

                            # backpropagate and update weights
                            loss.backward()
                            optimizer_encoder.step()

                # metrics
                running_loss_entropy += entropy_loss.item()
                running_loss_classifier1 += loss1.item()
                running_loss_classifier2 += loss2.item()
                running_loss_discrepancy += discrepancy_loss.item()
                
                # scheduler
                if cyclic_scheduler:
                    scheduler_encoder.step()
                    scheduler_classifiers.step()

            # get losses
            epoch_loss_entropy = running_loss_entropy / iters
            epoch_loss_classifier1 = running_loss_classifier1 / iters
            epoch_loss_classifier2 = running_loss_classifier2 / iters
            epoch_loss_discrepancy = running_loss_discrepancy / iters

            self.history['epoch_loss_entropy'].append(epoch_loss_entropy)
            self.history['epoch_loss_classifier1'].append(epoch_loss_classifier1)
            self.history['epoch_loss_classifier2'].append(epoch_loss_classifier2)
            self.history['epoch_loss_discrepancy'].append(epoch_loss_discrepancy)

            # self.evaluate on training data
            epoch_accuracy_source = self.evaluate(source_dataloader)
            epoch_accuracy_target = self.evaluate(target_dataloader)
            test_epoch_accuracy = self.evaluate(target_dataloader_test)

            self.history['accuracy_source'].append(epoch_accuracy_source)
            self.history['accuracy_target'].append(epoch_accuracy_target)

            # save checkpoint
            if test_epoch_accuracy > best_acc:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                            'classifier1_weights': self.classifier1.state_dict(),
                            'classifier2_weights': self.classifier2.state_dict()
                            }, save_path)
                best_acc = test_epoch_accuracy
                bad_epochs = 0
                
            else:
                bad_epochs += 1

            print('[Epoch {}/{}] entropy loss: {:.6f}; classifier 1 loss: {:.6f}; classifier 2 loss: {:.6f}; discrepancy loss: {:.6f}'.format(epoch+1, epochs, epoch_loss_entropy, epoch_loss_classifier1, epoch_loss_classifier2, epoch_loss_discrepancy))
            print('[Epoch {}/{}] accuracy source: {:.6f}; accuracy target: {:.6f}; val accuracy: {:.6f};'.format(epoch+1, epochs, epoch_accuracy_source, epoch_accuracy_target, test_epoch_accuracy))

            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier1.load_state_dict(best['classifier1_weights'])
        self.classifier2.load_state_dict(best['classifier2_weights'])

        return self.encoder, self.classifier1, self.classifier2

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
        self.classifier1.eval()
        self.classifier2.eval()

        labels_list = []
        outputs_list = []
        preds_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # predict
                features = self.encoder(data)
                outputs1 = self.classifier1(features)
                outputs2 = self.classifier2(features)
                
                outputs = F.softmax(outputs1 + outputs2, dim=1)

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
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list
            
        return accuracy

    def plot_metrics(self):
        """
        Plots the training metrics (only usable after calling .train()).
        """

        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=200)

        epochs = len(self.history['epoch_loss_entropy'])

        axs[0].plot(range(1, epochs+1), self.history['epoch_loss_entropy'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Entropy loss')

        axs[1].plot(range(1, epochs+1), self.history['epoch_loss_classifier1'], label='classifier 1')
        axs[1].plot(range(1, epochs+1), self.history['epoch_loss_classifier2'], label='classifier 2')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Classifier loss')
        axs[1].legend()

        axs[2].plot(range(1, epochs+1), self.history['epoch_loss_discrepancy'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Counts')
        axs[2].set_title('Discrepancy loss')      
            
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=200)

        axs[0].plot(range(1, epochs+1), self.history['accuracy_source'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_title('Accuracy on source')

        axs[1].plot(range(1, epochs+1), self.history['accuracy_target'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy on target')

        plt.show()

    @staticmethod
    def _weighted_crossentropy(input, target):
        input_softmax = F.softmax(input, dim=1)
        entropy = -input_softmax * torch.log(input_softmax + 1e-5) # standard info entropy with "anti-zero" term
        entropy = torch.sum(entropy, dim=1)

        weight = 1.0 + torch.exp(-entropy)
        weight = weight / torch.sum(weight).detach().item()

        return torch.mean(weight * F.cross_entropy(input, target, reduction='none'))

    @staticmethod
    def _entropy(input, epsilon=1e-5):
        # apply softmax
        input = F.softmax(input, dim=1)
        
        # entropy_condition
        entropy_condition = -input * torch.log(input + epsilon)
        entropy_condition = torch.sum(entropy_condition, dim=1).mean()
        
        # entropy_div
        input = torch.mean(input, 0) + epsilon
        entropy_div = input * torch.log(input)
        entropy_div = torch.sum(entropy_div)

        return entropy_condition + entropy_div

    @staticmethod
    def _discrepancy(input1, input2):
        return torch.mean(torch.abs(F.softmax(input1, dim=1) - F.softmax(input2, dim=1)))

    def _gradient_discrepancy(self, source_pack, target_pack):
        outputs1_source, outputs2_source, labels_source = source_pack
        outputs1_target, outputs2_target, labels_target = target_pack

        criterion = nn.CrossEntropyLoss()
        criterion_weighted = self._weighted_crossentropy

        gradient_loss = 0

        # get losses
        loss1_source = criterion(outputs1_source, labels_source)
        loss2_source = criterion(outputs2_source, labels_source)
        losses_source = [loss1_source, loss2_source]

        loss1_target = criterion_weighted(outputs1_target, labels_target)
        loss2_target = criterion_weighted(outputs2_target, labels_target)
        losses_target = [loss1_target, loss2_target]

        # get gradient loss from each classifier
        for classifier, loss_source, loss_target in zip([self.classifier1, self.classifier2], losses_source, losses_target):
            grad_cosine_similarity = []
            
            for name, params in classifier.named_parameters():
                real_grad = grad([loss_source], [params], create_graph=True, only_inputs=True, allow_unused=False)[0]
                fake_grad = grad([loss_target], [params], create_graph=True, only_inputs=True, allow_unused=False)[0]

                if len(params.shape) > 1:
                    cosine_similarity = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                else:
                    cosine_similarity = F.cosine_similarity(fake_grad, real_grad, dim=0)

                grad_cosine_similarity.append(cosine_similarity)

            # concatenate cosine similarities
            grad_cosine_similarity = torch.stack(grad_cosine_similarity)

            # get loss for this classifier
            gradient_loss += (1.0 - grad_cosine_similarity).mean()

        return gradient_loss/2.0 # mean of both gradient_loss(es)

    def _get_pseudo_labels(self, target_dataloader):
        self.encoder.eval()
        self.classifier1.eval()
        self.classifier2.eval()

        start_test = True

        with torch.no_grad():
            for data, labels in target_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # generate features
                features = self.encoder(data)

                outputs1 = self.classifier1(features)
                outputs2 = self.classifier2(features)
                outputs = outputs1 + outputs2

                if start_test:
                    all_features = features.float().cpu()
                    all_outputs = outputs.float().cpu()
                    all_labels = labels.float().cpu()
                    start_test = False
                
                else:
                    all_features = torch.cat((all_features, features.float().cpu()), dim=0)
                    all_outputs = torch.cat((all_outputs, outputs.float().cpu()), dim=0)
                    all_labels = torch.cat((all_labels, labels.float().cpu()), dim=0)

        all_outputs = F.softmax(all_outputs, dim=1)
        _, preds = torch.max(all_outputs, dim=1)
        accuracy = torch.sum(torch.squeeze(preds).float() == all_labels).item() / float(all_labels.size()[0])

        all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), dim=1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        # perform k-means clustering
        k = all_outputs.size(1)

        for i in range(2):
            if i == 0:
                aff = all_outputs.float().cpu().numpy()
            else:
                aff = np.eye(k)[preds_label]
            
            initial_centroid = aff.transpose().dot(all_features)
            initial_centroid = initial_centroid / (1e-8 + aff.sum(axis=0)[:,None])
            distance = cdist(all_features, initial_centroid, 'cosine')

            preds_label = distance.argmin(axis=1)
            accuracy_kmeans = np.sum(preds_label == all_labels.float().cpu().numpy()) / len(all_features)

        print('only source accuracy = {:.2f}% -> after clustering = {:.2f}%'.format(accuracy*100, accuracy_kmeans*100))
        return torch.tensor(preds_label, dtype=torch.long)