import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import random
from torch.utils import tensorboard
from lensing_envs.lensing_envs import Source
from cnn import CNN

class SersicDecoder(torch.nn.Module):
    """
    A wrapper around the a residual CNN with two heads, one for parameter optimisation and the other to stop addition of further Sérsics
    
    Attributes
    ----------

    image_shape: int
        Length of a single dimension of the square images to be studied
    hidden_dim: int
        Size of the hidden dimension before passing through to the heads
    output_dim: int
        Output dimension for the parameter optimisation head

    Methods
    -------
    forward(x)
        Feed-forward method
    """
    def __init__(self, image_shape, hidden_dim, output_dim=6):  # 6 = (n, r_e, q, θ, x₀, y₀)
        super().__init__()
        self.encoder = CNN(image_shape[0], image_shape[1], image_shape[2], hidden_dim)
        self.fc_sersic = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_stop = torch.nn.Linear(hidden_dim, 1)  # For termination probability

    def forward(self, x):  # x: (B, C, H, W)
        z = self.encoder(x)  # (B, hidden_dim)
        sersic_params = self.fc_sersic(z)  # (B, output_dim)
        stop_prob = torch.sigmoid(self.fc_stop(z)).squeeze(-1)  # (B,)
        return sersic_params, stop_prob


def strtobool(x):
    """
    Helper function to convert a string to a boolean value
    """
    if x.lower().strip() == 'true': return True
    else: return False

def parse_args():
    """
    Handles arguments for the argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the LR of the optimizer(s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=8e3,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, cuda will be enabled when possible')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    
    # Performance altering
    parser.add_argument('--num-steps', type=int, default=10,
                        help='number of steps per environment per rollout')
    args = parser.parse_args()
    return args 

selected_galaxies = np.load('selected_galaxies.npy')
selected_galaxies = np.mean(selected_galaxies, axis=-1, keepdims=True)
B, y, x, c = selected_galaxies.shape
selected_galaxies = np.reshape(selected_galaxies, (B, c, y, x))
selected_galaxies_min, selected_galaxies_max = selected_galaxies.min(axis=(-1,-2), keepdims=True), selected_galaxies.max(axis=(-1,-2), keepdims=True)
selected_galaxies = (selected_galaxies - selected_galaxies_min) / (selected_galaxies_max - selected_galaxies_min)

if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    print(f'[AGENT] Seed set to {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'runs/{args.exp_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
    def make_env(seed):
        def thunk():
            env = Source(hyperparameters={
                'B':47,
                'image_x':256,
                'image_y':256,
                'image_c':1,
                'seed':seed,
                'cuda':args.cuda,
            })
            return env
        return thunk
    
    env = make_env(seed=args.seed)()

    model = CNN(env.image_x, env.image_y, env.image_c, env.low.shape[1]).to(device)
    delta = 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    episodic_loss = 0

    temp = []
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_loss}') as progress:
        best_actions = None
        best_loss = np.inf

        for i in range(int(args.total_timesteps)):
            done = False
            state, _ = env.reset(selected_galaxies)

            params_list = []
            sersics = torch.zeros((args.num_steps, env.B, env.image_c, env.image_y, env.image_x)).to(device)
            j=0
            while j < args.num_steps:
                # gathering rollout data
                action = model(state)
                state, reward, done, _, info = env.step(action)
                sersics[j] = info['source']
                state = state.detach()
                params_list.append(action.detach().cpu().numpy())
                if torch.all(done):
                    break
                j += 1
            y_pred = torch.sum(sersics, dim=0)
            y_pred_flat = y_pred.view(env.B, -1)
            y_pred_min, _ = y_pred_flat.min(dim=-1, keepdim=True)
            y_pred_max, _ = y_pred_flat.max(dim=-1, keepdim=True)
            y_pred_min, y_pred_max = y_pred_min.view(env.B, 1, 1, 1), y_pred_max.view(env.B, 1, 1, 1)
            y_pred = (y_pred - y_pred_min) / (y_pred_max - y_pred_min + delta)
            y_labels = env.source_labels
            loss = torch.nn.functional.mse_loss(y_labels, y_pred)
            opt.zero_grad()
            loss.backward()
            opt.step()

            episodic_loss = loss.detach().cpu().numpy()
            if i > 4200: 
                temp.append(params_list)
            progress.set_description(f'episodic_loss: {episodic_loss}')
            progress.update()
            if args.log_train:
                writer.add_scalar("loss/actor_loss", episodic_loss, global_step=i)
            
            if episodic_loss < best_loss:
                best_loss = episodic_loss
                best_actions = np.array(params_list)
    np.save(f'best_actions_{args.exp_name}', best_actions)
    print('Best_loss',best_loss)