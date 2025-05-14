import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from datetime import datetime

# Create videos directory if it doesn't exist
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
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

def train_agent(env_name='CartPole-v1', episodes=1000, target_update=10, record_best=True):
    # Create training environment without rendering
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create a separate environment for recording video
    if record_best:
        video_path = os.path.join(VIDEOS_DIR, f"{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        record_env = gym.make(env_name, render_mode="rgb_array")
        record_env = RecordVideo(
            record_env,
            video_folder=video_path,
            episode_trigger=lambda x: x % 100 == 0  # Record every 100th episode
        )
    
    agent = DQNAgent(state_size, action_size)
    max_reward = float('-inf')
    
    # Reduce training frequency to save CPU
    train_frequency = 4  # Only train every 4 steps
    step_counter = 0
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            
            # Only train every few steps to reduce CPU usage
            step_counter += 1
            if step_counter % train_frequency == 0:
                loss = agent.replay()
            
            total_reward += reward
            state = next_state
            
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Track progress
        if total_reward > max_reward:
            max_reward = total_reward
            # Save the model when we get a new best reward
            torch.save(agent.policy_net.state_dict(), os.path.join(VIDEOS_DIR, f"{env_name}_best_model.pth"))
            
            if record_best and total_reward > 150:  # Only record if reward is good
                print(f"New best episode with reward {total_reward}! Recording video...")
                record_state = record_env.reset()[0]
                record_done = False
                while not record_done:
                    action = agent.act(record_state, training=False)
                    record_state, _, terminated, truncated, _ = record_env.step(action)
                    record_done = terminated or truncated
                record_env.close()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}, Max Reward: {max_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    if record_best:
        record_env.close()
    return agent

def evaluate_agent(agent, env, record=False):
    if record and isinstance(env, gym.Wrapper):
        # Environment is already wrapped for recording
        eval_env = env
    else:
        eval_env = env
    
    state = eval_env.reset()[0]
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    return total_reward

def load_trained_agent(env_name='CartPole-v1'):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    model_path = os.path.join(VIDEOS_DIR, f"{env_name}_best_model.pth")
    
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("Loaded pre-trained model")
    else:
        print("No pre-trained model found")
    
    return agent

if __name__ == "__main__":
    # Train a new agent
    agent = train_agent(episodes=500)  # Reduced episodes since CartPole learns faster
    
    # Or load a pre-trained agent
    # agent = load_trained_agent()
    
    # Record a final demonstration video
    video_path = os.path.join(VIDEOS_DIR, f"final_demonstration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    demo_env = gym.make('CartPole-v1', render_mode="rgb_array")
    demo_env = RecordVideo(demo_env, video_folder=video_path)
    
    print("Recording final demonstration...")
    final_reward = evaluate_agent(agent, demo_env, record=True)
    print(f"Final demonstration reward: {final_reward}")
    demo_env.close()
