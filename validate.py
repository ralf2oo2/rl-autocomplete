import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PPO import PPOAgent

max_seq_len = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

alphabet = list("abcdefghijklmnopqrstuvwxyz")
PAD_IDX = len(alphabet)  # 26
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}
vocab_size = len(alphabet) + 1  # +1 for PAD

agent = PPOAgent(nb_actions=vocab_size, device=device)
agent.load('./training/model1/ppo_checkpoint_epoch_8160.pth')

words = pd.read_csv("dataset_clean.csv")['word'].str.lower().tolist()
words = [w for w in words if w.isalpha()]

def validate(agent, words, num_samples=1000):
    char_correct = {}
    char_total = {}
    char_missed = {}
    correct_guesses = 0
    total_guesses = 0

    with torch.no_grad(): # might not be nessecary since the agent only trains when I call the update ppo function
        for _ in range(num_samples):
            word = random.choice(words)
            
            t = random.randint(1, len(word) - 1) # get random part of word, don't need to check for every part of the word

            partial = word[:t]
            target_char = word[t]

            # Encode partial wordc
            state = [char_to_int[c] for c in partial]
            pad_len = max_seq_len - len(state)
            state += [PAD_IDX] * pad_len
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)

            # Get action
            action, _, _ = agent.get_action_and_value(state_tensor)
            guessed_char = int_to_char.get(action.item(), '?')
            if guessed_char not in char_total:
                char_total[guessed_char] = 0
            char_total[guessed_char] += 1
            if guessed_char == target_char:
                correct_guesses += 1
                if guessed_char not in char_correct:
                    char_correct[guessed_char] = 0
                char_correct[guessed_char] += 1
            else:
                if target_char not in char_missed:
                    char_missed[target_char] = 0
                char_missed[target_char] += 1
            total_guesses += 1    

    accuracy = correct_guesses / total_guesses if total_guesses > 0 else 0
    return accuracy, correct_guesses, total_guesses, char_correct, char_missed, char_total

accuracy, correct_guesses, total_guesses, char_correct, char_missed, char_total = validate(agent, words, 1000)
print(f"Validation Accuracy: {accuracy:.4f} ({correct_guesses}/{total_guesses})")

all_chars = sorted(
    c for c in (set(char_total.keys()) | set(char_correct.keys()) | set(char_missed.keys()))
    if c != '?'
)
total_counts = [char_total.get(c, 0) for c in all_chars]
correct_counts = [char_correct.get(c, 0) for c in all_chars]
missed_counts = [char_missed.get(c, 0) for c in all_chars]

# Bar positions
x = np.arange(len(all_chars)) * 1.5  # Add space between character groups
width = 0.3

plt.figure(figsize=(10, 6))
plt.bar(x - width, total_counts, width, label='Total Guesses', color='skyblue')
plt.bar(x, correct_counts, width, label='Correct Guesses', color='seagreen')
plt.bar(x + width, missed_counts, width, label='Missed Guesses', color='lightcoral')

# Labels and formatting
plt.xlabel('Character')
plt.ylabel('Count')
plt.title('Character Prediction: Total, Correct, Missed Guesses')
plt.xticks(ticks=x, labels=all_chars)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()