import numpy as np

# Simple Grid World MDP
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.start = 0
        self.goal = size * size - 1
        self.rewards = np.zeros(self.n_states)
        self.rewards[self.goal] = 1  # reward at goal

    def step(self, state, action):
        row, col = divmod(state, self.size)
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        next_state = row * self.size + col
        reward = self.rewards[next_state]
        done = next_state == self.goal
        return next_state, reward, done

# Value Iteration for MDP
def value_iteration(env, gamma=0.9, theta=1e-6):
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)

    while True:
        delta = 0
        for s in range(env.n_states):
            if s == env.goal:
                continue
            v = V[s]
            action_values = []
            for a in range(env.n_actions):
                next_s, reward, _ = env.step(s, a)
                action_values.append(reward + gamma * V[next_s])
            V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, policy

# Run
env = GridWorld()
V, policy = value_iteration(env)

print("Value Function:")
print(V.reshape(4,4))
print("Policy (0:up, 1:down, 2:left, 3:right):")
print(policy.reshape(4,4))