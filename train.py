import torch
import random
import numpy as np
import pandas as pd
import os
from PPO import PPOAgent

save_dir = "./training/model2"

max_seq_len = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

alphabet = list("abcdefghijklmnopqrstuvwxyz")
PAD_IDX = len(alphabet)  # 26
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}
vocab_size = len(alphabet) + 1  # +1 for PAD

agent = PPOAgent(nb_actions=vocab_size, device=device)
#agent.load('./training/model1/ppo_checkpoint_epoch_80.pth')

words = pd.read_csv("dataset_clean.csv")['word'].str.lower().tolist()
words = [w for w in words if w.isalpha()]

def save(save_agent, dir, save_epoch):
    path = os.path.join(dir, f'ppo_checkpoint_epoch_{save_epoch}.pth')
    os.makedirs(dir, exist_ok=True)
    save_agent.save(path)
    print(f"saved model to {path}")

for epoch in range(100000000000000):
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    
    for _ in range(500):  # steps per epoch
        word = random.choice(words)
        for t in range(len(word) - 1):
            partial = word[:t]
            target_char = word[t]

            # Encode partial word
            state = [char_to_int[c] for c in partial]
            pad_len = max_seq_len - len(state)
            state += [PAD_IDX] * pad_len  # pad with PAD_IDX
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)  # shape: [1, max_seq_len]

            # Get action from PPO
            action, log_prob, value = agent.get_action_and_value(state_tensor)
            guessed_char = int_to_char.get(action.item(), '?')
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

    # Convert to arrays
    states = torch.tensor(states, dtype=torch.long)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    returns = np.array(returns)
    values = np.array(values)

    agent.ppo_update(states, actions, log_probs, returns, values)

    avg_reward = np.mean(rewards)

    if epoch % 20 == 0:
        save(agent, save_dir, epoch)
    print(f"Epoch {epoch}: Avg reward per step: {avg_reward:.2f}")

save(agent, save_dir, epoch)



