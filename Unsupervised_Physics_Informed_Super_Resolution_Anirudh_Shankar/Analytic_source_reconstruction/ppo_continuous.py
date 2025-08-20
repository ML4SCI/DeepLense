import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import argparse
from torch.utils import tensorboard
from tqdm import tqdm
import os
import random
import time
from lensing_envs.lensing_envs import Source
import cnn

def strtobool(x):
    """
    Helper function to convert a string to a boolean value
    """
    if x.lower().strip() == 'true': return True
    else: return False

# Actor module
class Actor(nn.Module):
    """
    Actor module for PPO

    Attributes
    ----------
    env: reinforcement learning environment

    Methods
    -------
    forward(X)
        Rollout feed-forward method
    forward_2(X)
        Post-rollout feed-forward method
    """
    def __init__(self, env):
        super().__init__()
        self.model = cnn.CNN(env.image_x, env.image_y, env.image_c, 2 * env.action_space.shape[1])
        self.out_shape = env.action_space.shape[1]
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [self.out_shape, self.out_shape], dim=-1)
        return means, log_stds.exp()
    def forward_2(self, X):
        X = X.view(args.num_steps * env.B, env.image_c, env.image_y, env.image_x)
        X = self.model(X)
        X = X.view(args.num_steps, env.B, 2 * self.out_shape)
        (means, log_stds) = torch.split(X, [self.out_shape, self.out_shape], dim=-1)
        return means, log_stds.exp()
    
# Critic module
class Critic(nn.Module):
    """
    Critic module for PPO

    Attributes
    ----------
    env: reinforcement learning environment

    Methods
    -------
    forward(X)
        Feed-forward method
    """
    def __init__(self, env):
        super().__init__()
        self.model = cnn.CNN(env.image_x, env.image_y, env.image_c, 1)
    
    def forward(self, X):
        return self.model(X)

def parse_args():
    """
    Handles arguments for the argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='Source-v0',
                        help='the id of the gym environment')
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
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, records videos of the agent\'s performance')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    
    # Performance altering
    parser.add_argument('--num-steps', type=int, default=10,
                        help='number of steps per environment per rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, LR is annealed')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, gae will not be computed')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the value of the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the value of the lambda parameter for gae')
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
                        help='number of iterations of policy updates')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, doesn\'t perform advantage normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help='the surrogate ratios\' clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, doesn\'t perform value loss clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='the value of the entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='the coefficient of the value function in the agent\'s loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
                        help='if and the threshold kl-d value with which early stopping must be evaluated')
    args = parser.parse_args()
    return args

def policy_loss(old_log_prob, log_prob, advantage, eps):
    """
    Computes the policy loss as the difference between the old and new policies, scaled by the advantages
    """
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage.unsqueeze(-1)
    
    m = torch.min(ratio*advantage.unsqueeze(-1), clipped)

    with torch.no_grad():
        logratio = log_prob - old_log_prob
        # old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    return -m, approx_kl, clipfracs

if __name__ == '__main__':
    args = parse_args()
    # run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    run_name = f'{args.gym_id}__{args.exp_name}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    print(f'[AGENT] Seed set to {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

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
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actor = Actor(env, activation=Mish).to(device)
    critic = Critic(env, activation=Mish).to(device)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )

    # selected_galaxies = np.load('1_sersic.npy')
    selected_galaxies = np.load('selected_galaxies.npy')
    selected_galaxies = np.mean(selected_galaxies, axis=-1, keepdims=True)
    B, y, x, c = selected_galaxies.shape
    selected_galaxies = np.reshape(selected_galaxies, (B, c, y, x))
    selected_galaxies_min, selected_galaxies_max = selected_galaxies.min(axis=(-1,-2), keepdims=True), selected_galaxies.max(axis=(-1,-2), keepdims=True)
    selected_galaxies = (selected_galaxies - selected_galaxies_min) / (selected_galaxies_max - selected_galaxies_min)
    start_time = time.time()
    update = 0
    episodic_reward = 0
    state_list = []
    print(f'[AGENT] Using {device}')
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        best_actions = None
        best_reward = -np.inf
        for i in range(int(args.total_timesteps)):
            update += 1
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.total_timesteps
                actor_lrnow = frac * args.learning_rate
                critic_lrnow = frac * args.learning_rate
                adam_actor.param_groups[0]['lr'] = actor_lrnow
                adam_critic.param_groups[0]['lr'] = critic_lrnow
            prev_logprob = None
            done = False
            state, _ = env.reset(selected_galaxies)
            state = torch.tensor(state, dtype=torch.float32).to(device)

            observations = torch.zeros((args.num_steps,)+selected_galaxies.shape, dtype=torch.float32).to(device)
            actions = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            logprobs = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            rewards = torch.zeros((args.num_steps, env.B,), dtype=torch.float32).to(device)
            dones = torch.zeros((args.num_steps, env.B,), dtype=torch.float32).to(device)
            values = torch.zeros((args.num_steps, env.B,), dtype=torch.float32).to(device)
            clip_fracs = []
            j = 0
            while j < args.num_steps:
                # gathering rollout data
                with torch.no_grad():
                    action_means, action_stds = actor(state)
                value = critic(state).flatten()
                dist = torch.distributions.Normal(action_means, action_stds)
                action = dist.sample()
                logprob = dist.log_prob(action)
                observations[j] = state
                actions[j] = action
                logprobs[j] = logprob
                state, reward, done, _, info = env.step(action.cpu().numpy())
                rewards[j] = torch.tensor(reward, dtype=torch.float32).to(device)
                dones[j] = torch.tensor(done, dtype=torch.float32).to(device)
                values[j] = value
                state = torch.from_numpy(state).float().to(device)
                if np.all(done):
                    break
                j += 1
            
            # advantage calculation
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            # done_index = dones.nonzero().max(axis=-1).item() if dones.any() else args.num_steps
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    advantages[t] = lastgaelam = rewards[t] + (1- dones[t]) * args.gamma * values[t] + args.gamma * args.gae_lambda * (1-dones[t]) * lastgaelam
                else:
                    advantages[t] = lastgaelam = rewards[t] + (1-dones[t+1]) * args.gamma * values[t+1] - (1-dones[t])*values[t] + args.gamma * args.gae_lambda * (1-dones[t+1]) * lastgaelam  

            action_means, action_stds = actor.forward_2(observations)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_logprobs = dist.log_prob(actions)
            actor_loss, approx_kl, clipfracs = policy_loss(logprobs, new_logprobs, advantages.detach(), args.clip_coef)
            actor_loss = actor_loss.mean()
            clip_fracs += clipfracs
            adam_actor.zero_grad()
            actor_loss.backward()
            adam_actor.step()

            critic_loss = advantages.pow(2).mean()
            adam_critic.zero_grad()
            critic_loss.backward()
            if args.log_train:
                writer.add_scalar("loss/actor_loss", actor_loss.detach(), global_step=i)
                writer.add_scalar("loss/advantage", advantages.detach().cpu().numpy().mean(), global_step=i)
                writer.add_scalar("reward/episode_reward", rewards.sum(dim=1).max().cpu().numpy(), global_step=i)
                writer.add_scalar("loss/critic_loss", critic_loss.detach(), global_step=i)
                writer.add_scalar('charts/approx_kl', approx_kl.item(), global_step=i)
                writer.add_scalar("charts/clipfrac", np.mean(clip_fracs), global_step=i)
                writer.add_scalar('charts/SPS', int(i/ (time.time() - start_time)), i)
            adam_critic.step()

            episodic_reward = rewards.sum(dim=1).max().cpu().numpy()
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()

            if episodic_reward > best_reward:
                best_reward = episodic_reward
                best_actions = actions.cpu().numpy()
    np.save(f'best_actions_{args.exp_name}', best_actions)