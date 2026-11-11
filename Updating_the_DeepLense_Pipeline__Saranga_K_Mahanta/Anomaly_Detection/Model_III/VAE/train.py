import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

sys.path.append("../")
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LOAD_PRETRAINED_MODEL,
    MODEL_PATH,
    SAVE_MODEL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from model import VAE, Decoder, Encoder
from train_dataloaders import create_dataloaders
from utils import device, set_seed, test_transforms, train_transforms


def train_epoch(model, dataloader, criterion, optimizer, scheduler, example_ct):
    model.train()
    train_losses = []
    train_kld_losses = []

    loop = tqdm(enumerate(dataloader), total=len(dataloader))

    for _batch_idx, img_batch in loop:
        X = img_batch.to(device)

        example_ct += len(img_batch)

        # forward prop
        y_pred, mu, var = model(X)

        # loss calculation
        loss = criterion(y_pred, X)
        kld_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        #         loss = ((y_pred - X)**2).sum() + model.encoder.kl

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # batch loss
        train_losses.append(loss.detach().cpu().numpy())
        train_kld_losses.append(kld_loss.detach().cpu().numpy())

    return model, np.mean(train_losses), np.mean(train_kld_losses), example_ct


def test_epoch(model, dataloader, criterion):
    model.eval()
    test_losses = []
    test_kld_losses = []

    y_pred_list = []
    y_truth_list = []

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=len(dataloader))

        for _batch_idx, img_batch in loop:
            X = img_batch.to(device)
            y_truth_list.append(X.detach().cpu().numpy())

            # forward prop
            y_pred, mu, var = model(X)

            y_pred_list.append(y_pred.detach().cpu().numpy())

            # loss and accuracy calculation
            loss = criterion(y_pred, X)
            kld_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
            #             loss = ((y_pred - X)**2).sum() + model.encoder.kl

            # batch loss and accuracy
            # print(f'Partial train loss: {loss.data}')
            test_losses.append(loss.detach().cpu().numpy())
            test_kld_losses.append(kld_loss.detach().cpu().numpy())

    return y_pred_list, y_truth_list, np.mean(test_losses), np.mean(test_kld_losses)


def plot_ae_outputs(model, dataloader, n=10):

    model.eval()
    plt.figure(figsize=(16, 4.5))

    img_batch = next(iter(dataloader))
    for i, img in enumerate(img_batch):
        if i >= n:
            break

        ax = plt.subplot(2, n, i + 1)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            rec_img, _, _ = model(img)

        img = img.permute(0, 2, 3, 1)
        plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original images")

        ax = plt.subplot(2, n, i + 1 + n)
        rec_img = rec_img.permute(0, 2, 3, 1)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed images")

    plt.draw()


def fit_model(model, checkpoint_path):

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        verbose=False,
    )
    criterion = nn.MSELoss()

    loss_dict = {
        "train_loss": [],
        "val_loss": [],
        "train_kld_loss": [],
        "val_kld_loss": [],
    }
    example_ct = 0  # number of examples seen
    min_val_loss = 999  # high value to initlialize

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        model, train_loss, train_kld_loss, example_ct = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, example_ct
        )
        _, _, val_loss, test_kld_loss = test_epoch(model, val_loader, criterion)
        #             scheduler.step()

        print(
            f"Train loss: {train_loss}, Val loss: {val_loss}, \n Train KLD: {train_kld_loss}, Val KLD: {test_kld_loss}"
        )

        if SAVE_MODEL and val_loss < min_val_loss:
            min_val_loss = val_loss
            print("New lower val loss. Saving model checkpoint!")
            torch.save(model.state_dict(), checkpoint_path)

        loss_dict["train_loss"].append(train_loss)
        loss_dict["val_loss"].append(val_loss)

        loss_dict["train_kld_loss"].append(train_kld_loss)
        loss_dict["val_kld_loss"].append(test_kld_loss)

        # if epoch % 10 == 0 or epoch + 1 == EPOCHS:
        plot_ae_outputs(model, val_loader, 10)

    return model, loss_dict


if __name__ == "__main__":
    set_seed(7)
    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, train_transforms, test_transforms, BATCH_SIZE
    )

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = VAE(encoder, decoder).to(device)

    if LOAD_PRETRAINED_MODEL:
        if device != "cpu":
            model.load_state_dict(torch.load(MODEL_PATH))
        else:
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )

    model, loss_dict = fit_model(model, checkpoint_path=MODEL_PATH)

    # Plot losses
    plt.figure(figsize=(19, 12))
    plt.semilogy(loss_dict["train_loss"], label="Train")
    plt.semilogy(loss_dict["val_loss"], label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    # plt.grid()
    plt.legend()
    plt.title("Reconstruction loss history")
    plt.savefig("Loss_history.png", dpi=80)
    plt.draw()

    # Plot KLD losses
    plt.figure(figsize=(19, 12))
    plt.semilogy(loss_dict["train_kld_loss"], label="Train")
    plt.semilogy(loss_dict["val_kld_loss"], label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Average KLD Loss")
    # plt.grid()
    plt.legend()
    plt.title("KLD Loss_history")
    plt.savefig("KLD_Loss_history.png", dpi=80)
    plt.show()
