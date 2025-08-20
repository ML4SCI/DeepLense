import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

class Source():
    """
    Analytic source construction module that creates compositions of Sérsics

    Arguments
    ---------
    hyperparameters: dict
        Dictionary of hyperparameters in a dictionary
    
    Methods
    -------
    reset(source_labels)
        Resets the environment with the given source labels and returns it as observations
    step(action, delta=1e-4)
        Steps the environment with the given action (Sérsic parameters) to return the new set of observations, the reward, and the done value
    _init_hyperparameters(hyperparameters)
        Initialises hyperparameters in the environment
    _process_actions(action)
        Scales the actions to match the environment's bounds
    _add_sersic(sersic_params)
        Helper function that adds Sérsics to the current collection, called when stepping the environment
    """
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        # r_e is the radius that encloses half of the total light of the galaxy, set here between 0 and 1 (scaled to image size)
        # n [0.1, 15], r_e [0, 1], q [0,1], theta [0, 2pi], center_x [0,1], center_y [0,1]
        self.low = torch.reshape(torch.tensor([0., 0., 0., 0., -1., -1.,]), (1, -1)) * torch.ones((self.B, 1))
        self.high = torch.reshape(torch.tensor([15., 1., 1., 2*torch.pi, 1., 1.]), (1, -1)) * torch.ones((self.B, 1))
        self.low, self.high = self.low.to(self.device), self.high.to(self.device)
        self.image_arcsec_bounds = torch.tensor([self.image_x * self.arcsec_per_pixel / 2, self.image_y * self.arcsec_per_pixel / 2])
        pos_x = torch.linspace(-self.image_arcsec_bounds[0], self.image_arcsec_bounds[0], self.image_x).to(self.device)
        pos_y = torch.linspace(-self.image_arcsec_bounds[1], self.image_arcsec_bounds[1], self.image_y).to(self.device)
        phi_y, phi_x = torch.meshgrid(pos_x, pos_y)
        self.phi_y, self.phi_x = torch.reshape(phi_y, (1, 1, self.image_y, self.image_x)) * torch.ones((self.B, self.image_c, self.image_y, self.image_x)).to(self.device), torch.reshape(phi_x, (1, 1, self.image_x, self.image_x)) * torch.ones((self.B, self.image_c, self.image_y, self.image_x)).to(self.device)
        self.phi_y, self.phi_x = self.phi_y.to(self.device), self.phi_x.to(self.device)

    def reset(self, source_labels):
        self.source_labels = torch.tensor(source_labels, device=self.device, dtype=torch.float32)
        self.constructed_sersic = torch.zeros(self.B, self.image_c, self.image_y, self.image_x).to(self.device)
        return self.source_labels, self._get_info()

    def step(self, action, delta=1e-4):
        action = self._process_actions(action)
        I = self._add_sersic(action)
        self.constructed_sersic += I
        source_diff = self.source_labels - self.constructed_sersic
        reward = -torch.mean((source_diff)**2, dim=(-3, -2, -1))
        reward_x = -torch.mean((torch.sum(source_diff, dim=-1)/self.image_x)**2, dim=(-2, -1))
        reward_y = -torch.mean((torch.sum(source_diff, dim=-2)/self.image_y)**2, dim=(-2, -1))
        reward += reward_x + reward_y
        done = torch.mean(torch.abs(source_diff), dim=(-3, -2, -1)) < delta
        if self.render_mode == 'source':
            return source_diff, reward, done, False, {'source':I, 'sersics':self.constructed_sersic, 'labels':self.source_labels}  
        return source_diff, 100*reward, done, False, {}
    
    def _init_hyperparameters(self, hyperparameters):
        self.B = 1
        self.image_x, self.image_y, self.image_c = 1, 1, 1
        self.seed = 0
        self.cuda = False
        self.arcsec_per_pixel = 0.001
        self.num_sersics = 10
        self.render_mode = 'source'
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available else 'cpu')
        print(f'[ENV] Using {self.device}')

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(f"[ENV] Seed set to {self.seed}")

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        # actions given in R
        squashed_action = torch.nn.functional.tanh(action) # in [-1,1]
        action = 0.5 * (squashed_action + 1) # in [0,1]
        action = torch.clip(action * (self.high - self.low) + self.low, self.low + 1e-1, self.high)
        return action
    
    def _add_sersic(self, sersic_params):
        delta = 1e-6
        def _sersic_law(R, r_e, b_n, n):
            return torch.exp(-b_n * ((R / (r_e + delta)) ** (1/(n + delta)) - 1))
        n, r_e_, q, theta, centers_x, centers_y = torch.tensor_split(sersic_params, 6, dim=-1) # (B, 1,)* on all
        centers_x, centers_y = centers_x*self.image_arcsec_bounds[0], centers_y*self.image_arcsec_bounds[1] # (B, 1)*
        b_n = 2 * n - 0.331 # (B, 1)*

        r_e = torch.linalg.norm(self.image_arcsec_bounds) * r_e_
        r_e = r_e.view(self.B, 1, 1, 1)
        q = q.view(self.B, 1, 1, 1)
        n = n.view(self.B, 1, 1, 1)
        theta = theta.view(self.B, 1, 1, 1)
        b_n = b_n.view(self.B, 1, 1, 1) # (B, 1, 1, 1) on all
        centers_x, centers_y = centers_x.view(self.B, 1, 1, 1), centers_y.view(self.B, 1, 1, 1)

        phi_y, phi_x = self.phi_y - centers_y, self.phi_x - centers_x # shifted coordinates in pixel scale (B, c, y, x,)*
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta) # (B, 1, 1, 1)*
        x_rot, y_rot = phi_x * cos_theta + phi_y * sin_theta, -phi_x * sin_theta + phi_y * cos_theta # rotated coordinates in (fractional) pixel scale (B, c, y, x)
        R = torch.sqrt((x_rot**2)/q + q*(y_rot**2)) * self.arcsec_per_pixel # (B, c, y, x,) in arcsec
        R[R==0] = 1e-6
        I = _sersic_law(R, r_e, b_n, n) # (B, c, y, x,)
        I_flat = I.view(self.B, -1)
        I_min, _ = I_flat.min(dim=-1, keepdim=True)
        I_max, _ = I_flat.max(dim=-1, keepdim=True)
        I_min, I_max = I_min.view(self.B, 1, 1, 1), I_max.view(self.B, 1, 1, 1)
        return (I - I_min) / (I_max - I_min + delta)