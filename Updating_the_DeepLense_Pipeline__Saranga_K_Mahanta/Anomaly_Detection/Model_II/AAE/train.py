import torch
import torch.nn as nn
import torch.optim as optim

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import device, set_seed, train_transforms, test_transforms
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, SAVE_MODEL
from train_dataloaders import create_dataloaders
from model import Encoder, Decoder, Discriminator


def train_epoch(model, dataloader, ae_criterion, optimizers, example_ct):
    encoder, decoder, disc = model["enc"], model["dec"], model["disc"]
    optim_encoder, optim_decoder, optim_disc, optim_encoder_reg = optimizers["optim_encoder"], optimizers["optim_decoder"], optimizers["optim_disc"], optimizers["optim_encoder_reg"]
    encoder.train()
    decoder.train()
    disc.train()
    
    rec_losses = []
    disc_losses = []
    gen_losses = []
    
    EPS = 1e-12
    
    loop = tqdm(enumerate(dataloader),total = len(dataloader))
    for batch_idx, img_batch in loop:

        X = img_batch.to(device)
        example_ct += len(img_batch)
        
        #update autoencoder weights via reconstruction loss
        encoding = encoder(X)
        fake = decoder(encoding)
        ae_loss = ae_criterion(fake, X)
        rec_losses.append(ae_loss.item())
        
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()
        ae_loss.backward()
        optim_encoder.step()
        optim_decoder.step()
        
        #update discriminator weights via cross-entropy loss
        z_real_gauss = (torch.randn(X.shape[0], 2048) * 5.).to(device)
        disc_real_gauss = disc(z_real_gauss)
        z_fake_gauss = encoder(X)
        disc_fake_gauss = disc(z_fake_gauss)
        
        disc_loss = -torch.mean(torch.log(disc_real_gauss + EPS) + torch.log(1- disc_fake_gauss + EPS))
        disc_losses.append(disc_loss.detach().cpu().numpy())
        
        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()
        
        #update generator/encoder via regularization loss
        z_fake_gauss = encoder(X)
        disc_fake_gauss = disc(z_fake_gauss)
        
        gen_loss = -torch.mean(torch.log(disc_fake_gauss + EPS))
        gen_losses.append(gen_loss.detach().cpu().numpy())
        
        optim_encoder_reg.zero_grad()
        gen_loss.backward()
        optim_encoder_reg.step()

    return model, np.mean(rec_losses), np.mean(disc_losses), np.mean(gen_losses), example_ct


def test_epoch(model, dataloader, criterion):

    encoder, decoder, disc = model["enc"], model["dec"], model["disc"]
    encoder.eval()
    decoder.eval()
    disc.eval()
    
    test_rec_losses = []
    y_pred_list = []
    y_truth_list = []

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader),total=len(dataloader))
        for batch_idx, img_batch in loop:
            X = img_batch.to(device)
            y_truth_list.append(X.detach().cpu().numpy())

            #forward prop
            encoded_space = encoder(X)
            recon = decoder(encoded_space)
            y_pred_list.append(recon.detach().cpu().numpy())

            #loss
            loss = criterion(recon, X).item() 

            #batch loss and accuracy
            test_rec_losses.append(loss)

    return y_pred_list, y_truth_list, np.mean(test_rec_losses)


def plot_ae_outputs(model, dataloader, n = 10):
    
    encoder, decoder, disc = model["enc"], model["dec"], model["disc"]
    encoder.eval()
    decoder.eval()
    disc.eval()
    plt.figure(figsize=(16,4.5))

    img_batch = next(iter(dataloader))
    for i, img in enumerate(img_batch):
        if i >= n:
            break

        ax = plt.subplot(2, n, i+1)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            encoded_space = encoder(img)
            rec_img = decoder(encoded_space)
            
        img = img.permute(0,2,3,1)
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
            
        ax = plt.subplot(2, n, i + 1 + n)
        rec_img = rec_img.permute(0,2,3,1)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Reconstructed images')
                      
    plt.draw()   


def fit_model(model, checkpoint_path):
        
        encoder, decoder, disc = model["enc"], model["dec"], model["disc"]
        #optimizers
        optim_encoder = optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
        optim_decoder = optim.Adam(decoder.parameters(), lr = LEARNING_RATE)
        optim_disc = optim.Adam(disc.parameters(), lr = LEARNING_RATE)
        optim_encoder_reg = optim.Adam(encoder.parameters(), lr = LEARNING_RATE * 0.1)
        optimizers = {
            'optim_encoder' : optim_encoder,
            'optim_decoder' : optim_decoder,
            'optim_disc' : optim_disc,
            'optim_encoder_reg' : optim_encoder_reg
        }
        
        ae_criterion = nn.MSELoss()
        loss_dict = {'train_rec_loss' : [],'val_rec_loss' : [], 'disc_loss' : [], 'gen_loss' : []}
        example_ct = 0  # number of examples seen
        min_val_loss = 999 #high value to initlialize
      
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            model, train_rec_loss, disc_loss, gen_loss, example_ct = train_epoch(model, train_loader, ae_criterion, optimizers, example_ct)
            _, _, val_rec_loss  = test_epoch(model, val_loader, ae_criterion)

            print(f'Train Recon loss: {train_rec_loss}, Val Recon loss: {val_rec_loss}, \n Disc loss: {disc_loss}, Gen loss: {gen_loss}\n')
            
            if SAVE_MODEL:
                if val_rec_loss < min_val_loss:
                    min_val_loss = val_rec_loss
                    print("New lower val loss. Saving model checkpoint!")
                    torch.save(encoder.state_dict(), checkpoint_path + 'encoder.pth')
                    torch.save(decoder.state_dict(), checkpoint_path + 'decoder.pth')
                    torch.save(disc.state_dict(), checkpoint_path + 'discriminator.pth')
    
            loss_dict['train_rec_loss'].append(train_rec_loss)
            loss_dict['val_rec_loss'].append(val_rec_loss)
            
            loss_dict['disc_loss'].append(disc_loss)
            loss_dict['gen_loss'].append(gen_loss)

            if epoch % 10 == 0 or epoch + 1 == EPOCHS:
                plot_ae_outputs(model, val_loader, 10)

        return model, loss_dict


if __name__ == "__main__":
    set_seed(7)
    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, 
        train_transforms, test_transforms, 
        BATCH_SIZE
        )

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    disc = Discriminator().to(device)
    model = {'enc':encoder,
            'dec': decoder,
            'disc': disc}
    model, loss_dict = fit_model(model, checkpoint_path = MODEL_PATH)

    # Plot losses
    plt.figure(figsize=(19,12))
    plt.semilogy(loss_dict['train_rec_loss'], label='Train')
    plt.semilogy(loss_dict['val_rec_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.savefig("Recon_Loss_history.png", dpi=80)  
    plt.draw()

    # Plot losses
    plt.figure(figsize=(19,12))
    plt.semilogy(loss_dict['disc_loss'], label='Discriminator loss')
    plt.semilogy(loss_dict['gen_loss'], label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average KLD Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.savefig("disc_vs_gen_loss_history.png", format="png", dpi=80)  
    plt.show()




