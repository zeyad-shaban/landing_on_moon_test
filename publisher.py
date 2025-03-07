from stable_baselines3 import PPO
from huggingface_sb3 import package_to_hub
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

model = PPO.load('./ppo-moon_lander.zip')
env_id = 'LunarLander-v3'

eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

package_to_hub(model=model, model_name='ppo_moon_lander', model_architecture='PPO', env_id=env_id, eval_env=eval_env, repo_id=f'zeyad-shaban/ppo-{env_id}', commit_message="init")2