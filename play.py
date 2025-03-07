import gymnasium as gym
import time
from stable_baselines3 import PPO

env = gym.make('LunarLander-v3', render_mode='human')
# model = PPO.load('./ppo-moon_lander.zip', env)
model = PPO.load('./ppo-moon_lander.zip')


while True:
    obs, _ = env.reset()
    R = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)

        # execute action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        R += reward

        # render environment
        time.sleep(0.01)
        env.render()

        # end episode logic
        if terminated or truncated:
            time.sleep(1)
            break

        obs = next_obs

    print(R)
