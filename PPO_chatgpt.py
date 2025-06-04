import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

    def get_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, old_log_probs, returns, advantages, clip_epsilon=0.2, epochs=4, batch_size=64):
    states = torch.stack(states)
    actions = torch.tensor(actions)
    old_log_probs = torch.tensor(old_log_probs)
    returns = torch.tensor(returns)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
            # Get current log probs and entropy
            logits = policy_net(batch_states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # Ratio for clipping
            ratios = torch.exp(log_probs - batch_old_log_probs)

            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            values = value_net(batch_states)
            value_loss = nn.MSELoss()(values, batch_returns)

            # Optimize policy
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Optimize value function
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()