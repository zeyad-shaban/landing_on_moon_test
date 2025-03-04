from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gymnasium as gym

eval_env = Monitor(gym.make('LunarLander-v3'))

model = PPO('MlpPolicy', eval_env)
model.load('./ppo-moon_lander.zip')

mean_reward, std_reward = evaluate_policy(model, eval_env, 100)

print(mean_reward, std_reward)