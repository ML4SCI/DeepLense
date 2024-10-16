
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from torch import nn
import torch.nn.functional as F

def plot(x_sampled,save_path=None):
    """ 
    input: output of model.decoder
    it plots them into a grid of 10x10
    """
    # x_sampled = x_sampled * 0.5 + 0.5  # n, nc, 128, 128
    grid = make_grid(x_sampled, nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1,2,0).detach().numpy())
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    

def sample( model = None, path=None, checkpoint = None, device=None, n=50,**kwargs):
    ''' 
    use: sample(checkpoint = "path/to/checkpoint.pth")
    # if checkpoint is None, model is used to sample
    '''
    if checkpoint is not None:
        model = BetaVAE()
        model.load_state_dict(torch.load(checkpoint))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        torch.manual_seed(0)
        z = torch.randn(n, model.latent_dimension).to(device) / 2
        
        x_sampled = model.decoder(z)
        x_sampled = x_sampled.cpu() # n, nc, 128, 128
        
        plot(x_sampled, save_path="plots/vae_generated")

def reconstruction(model = None,
                   checkpoint = None,
                   device=None, n=50,
                   trainset = None,
                   #valset = datasets_['val'],
                     **kwargs):
    ''' 
    use: reconstruction(checkpoint = "path/to/checkpoint.pth")
    # if checkpoint is None, model is used to sample
    
    n: number of images to reconstruct
    --> n//2 images from trainset
    --> n - n//2 images from valset
    
    '''
    if checkpoint is not None:
        model = BetaVAE()
        model.load_state_dict(torch.load(checkpoint))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        torch.manual_seed(0)
        #print()
        x_train = torch.stack([trainset[i] for i in range(n//2)])
        #print(x_train.shape)
       # x_val = torch.stack([valset[i][0] for i in range(n - n//2)])
        x = torch.cat([x_train],dim=0)#,x_val],dim=0)

        x = x.to(device)
        x_recon, mu, logvar = model(x)
        x_recon = x_recon.cpu() # n, nc, 128, 128
        
        plot(x_recon, save_path="plots/vae_recon", **kwargs)
        plot(x.cpu(), save_path="plots/vae_original", **kwargs)




def kl_with_standard(mu, logvar):

    batch_size = mu.size(0)
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    return klds.mean(1).mean(0 ,True)

def reconstruction_loss(x, x_recon, reduction='mean'):
    batch_size = x.size(0)
    return F.mse_loss(x_recon, x, reduction='mean')


class loss_beta(nn.Module):
    def __init__(self, beta=1.0,verbose=False):
        super().__init__()
        self.beta = beta
        self.verbose = verbose
    
    def forward(self, x, x_recon, mu, logvar):
        recon_loss = reconstruction_loss(x, x_recon)
        kl_loss = kl_with_standard(mu, logvar)
        if self.verbose:
            print("recon_loss: ", recon_loss)
            print("kl_loss: ", kl_loss)

            #inputs
            print("x: ", x)
            print("x_recon: ", x_recon)
            print("mu: ", mu)
            print("logvar: ", logvar)
            print("beta: ", beta)

        return recon_loss + self.beta*kl_loss#, recon_loss, kl_loss

#def loss_beta(x, x_recon, mu, logvar, ):

    

    


    