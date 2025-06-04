import torch
import random
import numpy as np
from torch.distributions import Categorical
import pandas as pd
from PPO import PPOAgent

max_seq_len = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

alphabet = list("abcdefghijklmnopqrstuvwxyz")
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_to_int.items()}
vocab_size = len(alphabet)

agent = PPOAgent(nb_actions=vocab_size, device=device)

words = pd.read_csv("dataset_clean.csv")['word'].str.lower().tolist()
words = [w for w in words if w.isalpha()]

for epoch in range(1000):
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    
    for _ in range(500):  # number of steps per epoch
        word = random.choice(words)
        for t in range(len(word) - 1):
            partial = word[:t]
            target_char = word[t]

            # Encode partial word
            state = [char_to_int[c] for c in partial]
            state = state + [0] * (max_seq_len - len(state))  # padding
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)

            # Get action from PPO
            action, log_prob, value = agent.get_action_and_value(state_tensor)
            guessed_char = int_to_char[action.item()]
            correct = guessed_char == target_char
            reward = 1 if correct else 0

            # Store experience
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(False)

        dones[-1] = True  # End of word is end of episode

    # Bootstrap value for last step
    next_state = torch.tensor([states[-1]], dtype=torch.long).to(device)
    _, _, next_value = agent.get_action_and_value(next_state)
    returns = agent.compute_gae(rewards, values, dones, next_value.item())

    # Convert to arrays for PPO
    states = np.array(states)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    returns = np.array(returns)
    values = np.array(values)

    agent.ppo_update(states, actions, log_probs, returns, values)

    if epoch % 10 == 0:
        avg_reward = np.mean(rewards)
        print(f"Epoch {epoch}: Avg reward per step: {avg_reward:.2f}")