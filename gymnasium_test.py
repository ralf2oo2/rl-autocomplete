import gymnasium as gym
from PPO import PPOAgent
import numpy as np
import torch

env = gym.make('CartPole-v1')
agent = PPOAgent(nb_actions=env.action_space.n)

max_episodes = 1000
max_steps = 500

for episode in range(max_episodes):
    state, _ = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    
    for _ in range(max_steps):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action, log_prob, value = agent.get_action_and_value(state_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


        states.append(state)
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        state = next_state
        if done:
            break

    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    _, _, next_value = agent.get_action_and_value(next_state_tensor)
    returns = agent.compute_gae(rewards, values, dones, next_value.item())

    agent.ppo_update(
        np.array(states),
        np.array(actions),
        np.array(log_probs),
        np.array(returns),
        np.array(values)
    )

    total_reward = sum(rewards)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")