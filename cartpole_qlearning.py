import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Discretize the state space for CartPole
def discretize_state(state, bins):
    cart_pos, cart_vel, pole_angle, pole_vel = state
    # Define bins
    cart_pos_bins = np.linspace(-4.8, 4.8, bins[0])
    cart_vel_bins = np.linspace(-3, 3, bins[1])  # approximate
    pole_angle_bins = np.linspace(-0.418, 0.418, bins[2])
    pole_vel_bins = np.linspace(-3, 3, bins[3])  # approximate

    # Digitize
    cart_pos_idx = np.digitize(cart_pos, cart_pos_bins) - 1
    cart_vel_idx = np.digitize(cart_vel, cart_vel_bins) - 1
    pole_angle_idx = np.digitize(pole_angle, pole_angle_bins) - 1
    pole_vel_idx = np.digitize(pole_vel, pole_vel_bins) - 1

    # Clip to valid indices
    cart_pos_idx = np.clip(cart_pos_idx, 0, bins[0]-1)
    cart_vel_idx = np.clip(cart_vel_idx, 0, bins[1]-1)
    pole_angle_idx = np.clip(pole_angle_idx, 0, bins[2]-1)
    pole_vel_idx = np.clip(pole_vel_idx, 0, bins[3]-1)

    return (cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx)

# Q-learning for CartPole
def q_learning_cartpole(episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, bins=(10,10,10,10)):
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n  # 2 actions

    # Q-table: shape (bins[0], bins[1], bins[2], bins[3], n_actions)
    q_table = np.zeros(bins + (n_actions,))

    rewards_per_episode = []

    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_state_obs, bins)
            done = terminated or truncated

            # Q-learning update (Bellman equation)
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action] * (not done)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()
    return q_table, rewards_per_episode

# Run Q-learning
q_table, rewards = q_learning_cartpole(episodes=1000)

# Plot rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning on CartPole')
plt.show()