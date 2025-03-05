import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from itertools import count
import torch.nn as nn
import os

env = make_vec_env('LunarLander-v3', 16)


class CustomNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer_dim=64):
        super().__init__()

        self.latent_dim_pi = last_layer_dim
        self.latent_dim_vf = last_layer_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, last_layer_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(last_layer_dim, out_channels)

        self.critic_head = nn.Linear(last_layer_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return features, features

    def forward_actor(self, x):
        return self.policy_head(x)

    def forward_critic(self, x):
        return self.critic_head(x)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.features_dim, self.action_space.n)


model = None
if os.path.exists('./ppo-moon_lander.zip'):
    model = PPO.load('./ppo-moon_lander.zip', env)
    model.verbose = 0
else:
    model = PPO(
        CustomPolicy,
        env,
        verbose=1,
        n_steps=1024,  # default 2048
        n_epochs=4,  # default 10
        gae_lambda=0.98,  # default 0.95
        ent_coef=0.02,  # default 0
        gamma=0.999,  # default 0.99
        batch_size=32,  # defalt 64
    )

eval_env = Monitor(gym.make('LunarLander-v3'))

for i in count(0):
    model.learn(total_timesteps=2e4)
    model.save('./ppo-moon_lander')

    reward_mean, reward_std = evaluate_policy(model, eval_env, 100)
    print(f'\n\n############### EPOCH: {i}\tmean: {reward_mean}\tstd: {reward_std}')


# no params: -356 -> -283 -> -266
# with all params:  -212 -> 10 -> -115
# batch size at 32: -187 -> 126 -> 72
