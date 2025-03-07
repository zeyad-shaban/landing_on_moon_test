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


model = None
if os.path.exists('./ppo-moon_lander.zip'):
    model = PPO.load('./ppo-moon_lander.zip', env)
    model.verbose = 0
else:
    model = PPO(
        'MlpPolicy',
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
