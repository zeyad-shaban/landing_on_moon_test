import gymnasium as gym
import pygame
import time

# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))  # Dummy window for input

env = gym.make('LunarLander-v3', render_mode='human')

def get_player_action():
    """
    Get action from keyboard input:
    - Nothing: Do nothing (0)
    - Left Arrow: Fire left engine (1)
    - Up Arrow: Fire main engine (2)
    - Right Arrow: Fire right engine (3)
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return 1
    elif keys[pygame.K_UP]:
        return 2
    elif keys[pygame.K_RIGHT]:
        return 3
    return 0

while True:
    obs, _ = env.reset()
    R = 0
    
    while True:
        # Handle pygame events to prevent window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                exit()

        # Get action from player
        action = get_player_action()

        # Execute action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        R += reward

        # Render environment
        time.sleep(0.01)
        env.render()

        if terminated or truncated:
            print(f"Episode Return: {R}")
            time.sleep(1)
            break

        obs = next_obs

    print(f"Final Score: {R}")