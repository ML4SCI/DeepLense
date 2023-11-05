import logging
import torch.nn as nn
from typing import Union, Any
import torch

def simsiam_train(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    use_lr_schedule: bool,
    scheduler: nn.Module,
    path: str,
    log_freq=100,
    ci=False
):
  logging.debug("Starting Training")
  for epoch in range(epochs):
      total_loss = 0    
      best_loss = float("inf")

      for batch_idx, batch in enumerate(train_loader):
          x0 = batch[0] 
          x1 = batch[1] 
          x0 = x0.to(device)
          x1 = x1.to(device)
          z0, p0 = model(x0)
          z1, p1 = model(x1)
          loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
          total_loss += loss.detach()
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()
          if ci:
              break
          

          if batch_idx % log_freq == 0:
              logging.debug(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

      if total_loss < best_loss:
          best_loss = total_loss
          torch.save(model.state_dict(), path)

      avg_loss = total_loss / len(train_loader)
      logging.debug(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
