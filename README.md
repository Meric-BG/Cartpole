# Cartpole RL Project

This project helps master Reinforcement Learning concepts from Sutton & Barto's book, focusing on chapter 2 (Multi-armed Bandits) and chapter 3 (Finite MDPs).

## Concepts Covered

### Chapter 2: Multi-armed Bandits ("8 Laws")
The key concepts from chapter 2 are implemented in `bandit_algorithms.py`:
1. **A k-armed bandit problem**: Decision making with multiple actions, each with unknown reward distribution.
2. **Action-value methods**: Estimating the value of each action.
3. **The 10-armed testbed**: Evaluating algorithms on a standard problem.
4. **Incremental implementation**: Updating estimates incrementally.
5. **Tracking a nonstationary problem**: Adapting to changing environments.
6. **Optimistic initial values**: Encouraging exploration with high initial estimates.
7. **Upper-Confidence-Bound (UCB) action selection**: Balancing exploration and exploitation.
8. **Gradient Bandit Algorithms**: Using gradient ascent to learn preferences.

### Chapter 3: Finite MDPs and Bellman Equation
- **MDP**: Markov Decision Process with states, actions, transitions, rewards.
- **Bellman Equation**: Recursive relationship for value functions:
  - State-value: V(s) = max_a [R(s,a) + γ ∑ P(s'|s,a) V(s')]
  - Action-value: Q(s,a) = R(s,a) + γ ∑ P(s'|s,a) max_a' Q(s',a')

## Files

- `cartpole_random.py`: Basic CartPole with random actions.
- `cartpole_qlearning.py`: Q-learning on discretized CartPole (Bellman equation in action).
- `gridworld_mdp.py`: Value iteration on a simple grid world MDP.
- `bandit_algorithms.py`: Implementation of bandit algorithms from chapter 2.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run scripts with Python.

## Next Steps

- Experiment with different hyperparameters.
- Implement policy iteration.
- Try function approximation for continuous states.