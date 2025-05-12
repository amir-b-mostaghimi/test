import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Create the CartPole environment
env = gym.make('CartPole-v1')
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")

states = env.observation_space.shape[0]
actions = env.action_space.n

# Create the deep learning model
model = Sequential()
model.add(Dense(24, activation='relu', input_dim=states))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))
model.summary()

# Build the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(
    model=model,
    memory=memory,
    policy=policy,
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    enable_double_dqn=True
)

# Compile the agent
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
print("Training the agent...")
history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Test the trained agent
print("\nTesting the agent...")
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(f"Mean reward over 100 episodes: {np.mean(scores.history['episode_reward'])}")

# Visualize some episodes
print("\nVisualizing 5 episodes...")
_ = dqn.test(env, nb_episodes=5, visualize=True)

env.close()
