from .supervised import Supervised
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Adda(Supervised):
    """
    Paper: Adversarial Discriminative Domain Adaptation
    Authors: Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell
    """

    def __init__(self, source_encoder, target_encoder, classifier, discriminator):
        """
        Arguments:
        ----------
        source_encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.
            Must have been previously trained in a supervised manner on the source dataset.

        target_encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.
            Must have the same architecture as `source_encoder`.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.

        discriminator: PyTorch neural network
            Neural network that receives an array of size X and classifies it into 2 classes.
            It discriminates between encodings of the source domain and the target domain.
        """

        super().__init__(target_encoder, classifier)

        self.discriminator = discriminator.to(self.device)

        # we consider self.encoder = target_encoder
        self.source_encoder = source_encoder.to(self.device)
        self.encoder.load_state_dict(self.source_encoder.state_dict())

        # disbale grad in already trained networks
        for param in self.source_encoder.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = False

    def train(self, source_dataloader, target_dataloader, target_dataloader_test, epochs, hyperparams, save_path):
        """
        Trains the model (encoder + classifier) and discriminator.

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
        lr_target = hyperparams['learning_rate_target']
        lr_discriminator = hyperparams['learning_rate_discriminator']
        wd = hyperparams['weight_decay']
        cyclic_scheduler = hyperparams['cyclic_scheduler']
        
        iters = max(len(source_dataloader), len(target_dataloader))
        
        self.source_encoder.eval()
        self.classifier.eval()
        
        # configure optimizers and schedulers
        optimizer_target = optim.Adam(self.encoder.parameters(), lr=lr_target, weight_decay=wd)
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator, weight_decay=wd)

        if cyclic_scheduler:
            scheduler_target = optim.lr_scheduler.OneCycleLR(optimizer_target, lr_target, epochs=epochs, steps_per_epoch=iters)
            scheduler_discriminator = optim.lr_scheduler.OneCycleLR(optimizer_discriminator, lr_discriminator, epochs=epochs, steps_per_epoch=iters)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 15
        bad_epochs = 0

        self.history = {'loss_discriminator': [], 'loss_encoder': [], 'accuracy': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss_discriminator = 0.0
            running_loss_target = 0.0
            
            # set network to training mode
            self.discriminator.train()
            self.encoder.train()

            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for (data_source, _), (data_target, _) in zip(source_dataloader, target_dataloader):
                data_source = data_source.to(self.device)
                data_target = data_target.to(self.device)

                # useful when creating labels for self.discriminator
                batch_size_source = data_source.size(0)
                batch_size_target = data_target.size(0)

                # train self.discriminator
                ## zero gradients
                optimizer_discriminator.zero_grad()

                ## encode features and concatenate them
                features_source = self.source_encoder(data_source)
                features_target = self.encoder(data_target)

                ## generate real and fake labels for self.discriminator
                label_source = Variable(torch.zeros(batch_size_source, dtype=torch.long)).to(self.device)
                label_target = Variable(torch.ones(batch_size_target, dtype=torch.long)).to(self.device)
                label_concat = torch.cat([label_source, label_target], dim=0)

                ## classify with self.discriminator
                outputs_source = self.discriminator(features_source)
                outputs_target = self.discriminator(features_target)
                outputs_concat = torch.cat([outputs_source, outputs_target], dim=0)

                ## get loss for self.discriminator
                loss_discriminator = criterion(outputs_concat, label_concat)

                ## backpropagate and update weights
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # train target encoder
                ## zero gradients
                optimizer_discriminator.zero_grad()
                optimizer_target.zero_grad()

                ## encode target features
                features_target = self.encoder(data_target)

                ## classify encoding with self.discriminator
                outputs_target = self.discriminator(features_target)

                ## get loss for target encoder
                label_target = Variable(torch.zeros(batch_size_target, dtype=torch.long)).to(self.device)
                loss_encoder = criterion(outputs_target, label_target)
                
                ## backpropagate and update weights
                loss_encoder.backward()
                optimizer_target.step()

                # metrics
                running_loss_discriminator += loss_discriminator.item()
                running_loss_target += loss_encoder.item()
                
                # scheduler step
                if cyclic_scheduler:
                    scheduler_discriminator.step()
                    scheduler_target.step()

            # get losses
            epoch_loss_discriminator = running_loss_discriminator / iters
            epoch_loss_target = running_loss_target / iters
            self.history['loss_discriminator'].append(epoch_loss_discriminator)
            self.history['loss_encoder'].append(epoch_loss_target)

            # self.evaluate on testing data (target domain)
            epoch_accuracy = self.evaluate(target_dataloader)
            test_epoch_accuracy = self.evaluate(target_dataloader_test)
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
                
            print('[Epoch {}/{}] discriminator loss: {:.6f}; target loss: {:.6f}; accuracy target: {:.6f}; val accuracy: {:.6f};'.format(epoch+1, epochs, epoch_loss_discriminator, epoch_loss_target, epoch_accuracy, test_epoch_accuracy))
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])
        
        return self.encoder, self.classifier

    def plot_metrics(self):
        """
        Plots the training metrics (only usable after calling .train()).
        """

        # plot metrics from target
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=200)

        epochs = len(self.history['loss_discriminator'])

        axs[0].plot(range(1, epochs+1), self.history['loss_discriminator'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Target discriminator loss')

        axs[2].plot(range(1, epochs+1), self.history['loss_encoder'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Loss')
        axs[2].set_title('Target encoder loss')      

        axs[1].plot(range(1, epochs+1), self.history['accuracy'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Target accuracy')
            
        plt.show()