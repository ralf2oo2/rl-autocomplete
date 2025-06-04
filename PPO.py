import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
class ActorCritic(nn.Module):

    def __init__(self, nb_actions, vocab_size=26, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, nb_actions)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        h = h.squeeze(0)

        h = self.head(h)
        return self.actor(h), self.critic(h)

class PPOAgent:
    def __init__(self, nb_actions, device='cpu', gamma=0.99, lam=0.95, clip_eps=0.2, lr=2.5e-4, epochs=4, batch_size=64):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(nb_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action_and_value(self, state):
        logits, value = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def compute_gae(self, rewards, values, dones, next_value):
        values = np.append(values, next_value)
        gae = 0
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
        return returns

    def ppo_update(self, states, actions, log_probs, returns, values):
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for i in range(0, len(states), self.batch_size):
                idx = indices[i:i+self.batch_size]
                s_batch = torch.tensor(states[idx], dtype=torch.float32).to(self.device)
                a_batch = torch.tensor(actions[idx]).to(self.device)
                old_log_probs_batch = torch.tensor(log_probs[idx]).to(self.device)
                ret_batch = torch.tensor(returns[idx], dtype=torch.float32).to(self.device)
                adv_batch = torch.tensor(advantages[idx], dtype=torch.float32).to(self.device)

                logits, value = self.model(s_batch)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(a_batch)

                ratio = (new_log_probs - old_log_probs_batch).exp()
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), ret_batch)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()