from pycarrot.envs import GymEnvironment
from pycarrot.agents import DQNAgent

# Create a Gym environment
env = GymEnvironment("CartPole-v1")

# Define the agent with default settings
agent = DQNAgent(env=env, discount_factor=0.99, epsilon=0.1)

# Train the agent
agent.train(episodes=500)

# Evaluate the agent
reward = agent.evaluate(episodes=10)
print(f"Average Reward: {reward}")

import matplotlib.pyplot as plt

# Plot rewards over episodes
plt.plot(agent.history['rewards'])
plt.title("Training Progress")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()