import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to get the initial state
state, info = env.reset()

# Run for a few episodes with random actions
for episode in range(5):
    state, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        # Sample a random action (0: left, 1: right)
        action = env.action_space.sample()
        # Take the action and observe the next state, reward, etc.
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()