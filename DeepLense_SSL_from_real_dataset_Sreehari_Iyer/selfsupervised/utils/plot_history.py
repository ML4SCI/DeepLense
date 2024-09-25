import torch
import os
import matplotlib.pyplot as plt
import numpy as np



def plot_history(ckpt_path: str, plt_save_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    loss = state['history']['loss_epochwise']
    loss_stepwise = state['history']['loss_stepwise']
    steps_per_epoch = len(loss_stepwise) / len(loss)
    knntop1 = [top1 for top1 in state['history']['knn_top1'] if top1 is not None]
    
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(15, 6))
    
    x = np.arange(0, len(loss))
    axs.plot(x, loss)
    axs.grid(alpha = 0.5)
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    axs.set_title("Loss vs Epoch")
    
    plt.show()
    plt.savefig(plt_save_path)


