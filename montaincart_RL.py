import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.99,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize policy network
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent(env_name='MountainCar-v0', episodes=2000, target_update=5):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create videos directory if it doesn't exist
    videos_dir = "best_episodes"
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    
    agent = DQNAgent(state_size, action_size)
    max_reward = -100
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            modified_reward = reward + 10 * next_state[0]
            modified_reward = np.clip(modified_reward, -1.0, 1.0)
            
            agent.remember(state, action, modified_reward, next_state, done)
            loss = agent.replay()
            
            total_reward += modified_reward  # Track original reward for logging
            state = next_state
            
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Track progress
        max_reward = max(max_reward, total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Max Reward: {max_reward}, Epsilon: {agent.epsilon:.3f}")
        
        # Record best episodes
        if total_reward >= max_reward and total_reward > -100:
            print(f"New best episode with reward {total_reward}! Recording video...")
            video_env = gym.make(env_name, render_mode='rgb_array')
            video_env = RecordVideo(
                video_env, 
                videos_dir, 
                episode_trigger=lambda x: True,
                name_prefix=f"best_episode_reward_{total_reward:.0f}"
            )
            test_reward = evaluate_agent(agent, video_env)
            print(f"Test episode finished with reward {test_reward}")
            video_env.close()
    
    env.close()
    return agent

def evaluate_agent(agent, env):
    state = env.reset()[0]
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    return total_reward

# Run baby run
if __name__ == "__main__":
    agent = train_agent()
