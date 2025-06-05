import torch
import os
from PPO import PPOAgent  # assumes PPOAgent is implemented in PPO.py

# Constants
max_seq_len = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Character mappings
alphabet = list("abcdefghijklmnopqrstuvwxyz")
PAD_IDX = len(alphabet)  # 26
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}

# Load the trained agent
agent = PPOAgent(nb_actions=len(alphabet) + 1, device=device)
agent.load('./training/model1/ppo_checkpoint_epoch_8160.pth')  # path to saved checkpoint with highest count

def predict_next_char(partial_word: str) -> str:
    partial_word = partial_word.lower()
    encoded = [char_to_int.get(c, PAD_IDX) for c in partial_word if c in char_to_int]
    if len(encoded) > max_seq_len:
        encoded = encoded[-max_seq_len:]  # keep only last max_seq_len characters
    else:
        encoded += [PAD_IDX] * (max_seq_len - len(encoded))  # pad to fixed length

    state_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    action, _, _ = agent.get_action_and_value(state_tensor)

    return int_to_char.get(action.item(), '')

if __name__ == "__main__":
    while True:
        user_input = input("Enter partial word (or 'quit'): ").strip()
        if user_input.lower() == 'quit':
            break
        predicted_char = predict_next_char(user_input)
        print(f"Predicted next character: {predicted_char}")