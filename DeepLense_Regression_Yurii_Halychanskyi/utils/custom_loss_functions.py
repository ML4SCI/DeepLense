import torch

def mse_loss_wgtd(pred, true, wgt=1.):
  loss = wgt*(pred-true).pow(2)
  return loss.mean()

def root_mean_squared_error(p, y): 
    return torch.sqrt(mse_loss_wgtd(p.view(-1), y.view(-1)))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs()
    return loss.mean()