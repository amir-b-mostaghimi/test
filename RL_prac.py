import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Neural Network for Policy
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

# Training setup
env = gym.make('CartPole-v1')  # Remove render_mode for faster training
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.001)  # Lower learning rate
gamma = 0.99  # Discount factor
eps = 0.1  # Exploration rate for entropy bonus

# Training loop
max_reward = 0
for episode in range(1000):
    state = env.reset()[0]
    log_probs = []
    rewards = []
    entropies = []
    done = False
    
    # Collect trajectory
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()  # Add entropy for exploration
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        state = next_state
    
    # Calculate discounted rewards
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    # Normalize returns for stability
    if len(returns) > 1:  # Only normalize if we have enough samples
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Calculate loss and update policy
    policy_loss = []
    entropy_loss = []
    for log_prob, R, entropy in zip(log_probs, returns, entropies):
        policy_loss.append(-log_prob * R)
        entropy_loss.append(-entropy)  # Encourage exploration
    
    optimizer.zero_grad() # reset the optimizer
    policy_loss = torch.stack(policy_loss).sum()
    entropy_loss = torch.stack(entropy_loss).sum()
    loss = policy_loss + eps * entropy_loss  # Add entropy bonus
    loss.backward()
    
    # Gradient clipping to prevent too large updates
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    
    # Track progress
    total_reward = sum(rewards)
    max_reward = max(max_reward, total_reward)
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Max Reward: {max_reward}")
    
    # Render best episodes
    if total_reward >= max_reward and total_reward > 100:
        print(f"New best episode with reward {total_reward}!")
        # Render a test episode with the current policy
        test_env = gym.make('CartPole-v1', render_mode='human')
        state = test_env.reset()[0]
        done = False
        test_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()  # Greedy action
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            test_reward += reward
        test_env.close()
        print(f"Test episode finished with reward {test_reward}")

env.close()
