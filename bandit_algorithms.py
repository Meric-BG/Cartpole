import numpy as np
import matplotlib.pyplot as plt

# Multi-armed Bandit
class Bandit:
    def __init__(self, k=10, true_q=None):
        self.k = k
        if true_q is None:
            self.true_q = np.random.normal(0, 1, k)
        else:
            self.true_q = true_q
        self.optimal_action = np.argmax(self.true_q)

    def pull(self, action):
        return np.random.normal(self.true_q[action], 1)

# Epsilon-Greedy
def epsilon_greedy(bandit, steps=1000, epsilon=0.1):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []
    optimal_actions = []

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(bandit.k)
        else:
            action = np.argmax(Q)
        reward = bandit.pull(action)
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.optimal_action else 0)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

    return rewards, optimal_actions

# Upper-Confidence-Bound (UCB)
def ucb(bandit, steps=1000, c=2):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []
    optimal_actions = []

    for t in range(1, steps + 1):
        ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-5))
        action = np.argmax(ucb_values)
        reward = bandit.pull(action)
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.optimal_action else 0)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

    return rewards, optimal_actions

# Gradient Bandit
def gradient_bandit(bandit, steps=1000, alpha=0.1):
    H = np.zeros(bandit.k)
    rewards = []
    optimal_actions = []
    baseline = 0

    for t in range(steps):
        probs = np.exp(H) / np.sum(np.exp(H))
        action = np.random.choice(bandit.k, p=probs)
        reward = bandit.pull(action)
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.optimal_action else 0)

        baseline += (reward - baseline) / (t + 1)
        for a in range(bandit.k):
            if a == action:
                H[a] += alpha * (reward - baseline) * (1 - probs[a])
            else:
                H[a] -= alpha * (reward - baseline) * probs[a]

    return rewards, optimal_actions

# Run and compare
bandit = Bandit()
eps_rewards, eps_opt = epsilon_greedy(bandit)
ucb_rewards, ucb_opt = ucb(bandit)
grad_rewards, grad_opt = gradient_bandit(bandit)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(eps_rewards) / np.arange(1, len(eps_rewards)+1), label='ε-greedy')
plt.plot(np.cumsum(ucb_rewards) / np.arange(1, len(ucb_rewards)+1), label='UCB')
plt.plot(np.cumsum(grad_rewards) / np.arange(1, len(grad_rewards)+1), label='Gradient')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(eps_opt) / np.arange(1, len(eps_opt)+1), label='ε-greedy')
plt.plot(np.cumsum(ucb_opt) / np.arange(1, len(ucb_opt)+1), label='UCB')
plt.plot(np.cumsum(grad_opt) / np.arange(1, len(grad_opt)+1), label='Gradient')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.show()