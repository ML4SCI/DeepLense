import torch
import torch.nn as nn
from typing import Any, Union
import logging


def finetune(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    saved_model_path: str,
    valid_loader: Any,
    scheduler=None,
    ci=False,
):
    best_loss = float("inf")
    best_accuracy = float("-inf")
    all_val_loss = []
    all_val_accuracy = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(
            train_loader
        ):  # for batch_idx, batch in enumerate(train_loader):
            img1 = batch[0].to(device)
            label = batch[-1].to(device)
            optimizer.zero_grad()
            output = model(img1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss
            if ci:
                break

            if batch_idx % 100 == 0:
                logging.debug(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
                )

        epoch_loss = epoch_loss / len(train_loader)

        with torch.no_grad():
            logging.debug("====== Eval started ======")
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for batch_idx, batch in enumerate(
                valid_loader
            ):  # for batch_idx, batch in enumerate(train_loader):
                data = batch[0].to(device)
                label = batch[-1].to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
                if ci:
                    break

            epoch_val_accuracy = epoch_val_accuracy / len(valid_loader)

            if epoch_val_accuracy > best_accuracy:
                best_accuracy = epoch_val_accuracy
                torch.save(model.state_dict(), saved_model_path)
                logging.debug("====== Model saved ======")

            epoch_val_loss = epoch_val_loss / len(valid_loader)
            all_val_loss.append(epoch_val_loss)

        all_val_accuracy.append(epoch_val_accuracy.item() * 100)

        logging.debug(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} \n"
        )


def finetune_regression(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    saved_model_path: str,
    valid_loader: Any,
    scheduler=None,
    ci=False,
):
    best_loss = float("inf")
    best_accuracy = float("-inf")
    all_val_loss = []
    all_val_accuracy = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(
            train_loader
        ):  # for batch_idx, batch in enumerate(train_loader):
            img1 = batch[0].to(device)
            label = batch[-1].to(device)
            optimizer.zero_grad()
            output = model(img1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss
            if ci:
                break

            if batch_idx % 100 == 0:
                logging.debug(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
                )

        epoch_loss = epoch_loss / len(train_loader)

        with torch.no_grad():
            logging.debug("====== Eval started ======")
            model.eval()
            epoch_val_loss = 0
            for batch_idx, batch in enumerate(valid_loader):
                data = batch[0].to(device)
                label = batch[-1].to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                epoch_val_loss += val_loss
                if ci:
                    break

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), saved_model_path)
                logging.debug("====== Model saved ======")

            epoch_val_loss = epoch_val_loss / len(valid_loader)
            all_val_loss.append(epoch_val_loss)

        logging.debug(
            f"Epoch : {epoch+1} - loss : {epoch_loss} - val_loss : {epoch_val_loss} \n"
        )
