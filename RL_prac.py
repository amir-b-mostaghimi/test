import gym
import numpy as np

# Create the environment
env = gym.make('CartPole-v1', render_mode='human')

# Reset the environment
reset_result = env.reset()
# Handle both old and new Gym API formats
if isinstance(reset_result, tuple):
    observation = reset_result[0]  # New API (obs, info)
else:
    observation = reset_result  # Old API (just obs)

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    
    # Handle step() return values for both old and new Gym API
    step_result = env.step(action)
    
    if len(step_result) == 5:  # New API: obs, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:  # Old API: obs, reward, done, info
        observation, reward, done, info = step_result
    
    if done:
        # Reset the environment
        reset_result = env.reset()
        # Handle both old and new Gym API formats
        if isinstance(reset_result, tuple):
            observation = reset_result[0]
        else:
            observation = reset_result

env.close()

