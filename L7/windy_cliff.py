import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

np.random.seed(0)

class WindyCliffWorld(gym.Env):
    def __init__(self):
        super(WindyCliffWorld, self).__init__()
        
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.goal_state = (3, 9)
        self.cliff = [(3, i) for i in range(1, 9)]
        self.obstacles = [(2, 4), (4, 4), (2, 7), (4, 7)]
        
        self.wind_strength = {
            (i, j): np.random.choice([-1, 0, 1]) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        }

        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        
        self.state = self.start_state
        
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)
    
    def step(self, action):
        new_state = (self.state[0] + self.action_effects[action][0], self.state[1] + self.action_effects[action][1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        # Apply wind effect
        wind = self.wind_strength[new_state]
        new_state = (new_state[0] + wind, new_state[1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        if new_state in self.cliff:
            reward = -100
            done = True
            new_state = self.start_state
        elif new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False

        self.state = new_state
        return self.state_to_index(new_state), reward, done, {}
    
    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.state] = 1  # Current position
        for c in self.cliff:
            grid[c] = -1  # Cliff positions
        for o in self.obstacles:
            grid[o] = -0.5  # Obstacle positions
        grid[self.goal_state] = 2  # Goal position
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        fig.canvas.draw()
        plt.close(fig)
        image = np.array(fig.canvas.renderer.buffer_rgba())
        return image

# Create and register the environment
env = WindyCliffWorld()

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])
            
            state = next_state
            step_count += 1
        
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed")
    
    return q_table, episode_rewards

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        while not done and step_count < max_steps:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])
            
            state = next_state
            action = next_action
            step_count += 1
        
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed")
    
    return q_table, episode_rewards

def save_gif(frames, path='./', filename='gym_animation.gif'):
    imageio.mimsave(os.path.join(path, filename), frames, duration=0.1)  # Faster animation

def visualize_policy(env, q_table, filename='q_learning.gif'):
    state = env.reset()
    frames = []
    done = False
    step_count = 0
    max_steps = 100  # Limit steps for visualization
    
    print(f"Visualizing policy for {filename}...")
    while not done and step_count < max_steps:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        frames.append(env.render())
        step_count += 1
    
    if step_count >= max_steps:
        print(f"Warning: Policy visualization for {filename} reached max steps without reaching goal")
    
    save_gif(frames, filename=filename)
    print(f"Saved {filename}")

def run_hyperparameter_experiments():
    print("Starting hyperparameter experiments...")
    num_episodes = 500
    gamma = 0.99
    
    # Define different values for alpha and epsilon
    alphas = [0.1, 0.5]
    epsilons = [0.1, 0.5]
    
    # Store results
    q_learning_results = {}
    sarsa_results = {}
    
    # Run Q-learning experiments
    for alpha in alphas:
        for epsilon in epsilons:
            print(f"Running Q-learning with α={alpha}, ε={epsilon}")
            env = WindyCliffWorld()
            _, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)
            q_learning_results[(alpha, epsilon)] = rewards
    
    # Run SARSA experiments
    for alpha in alphas:
        for epsilon in epsilons:
            print(f"Running SARSA with α={alpha}, ε={epsilon}")
            env = WindyCliffWorld()
            _, rewards = sarsa(env, num_episodes, alpha, gamma, epsilon)
            sarsa_results[(alpha, epsilon)] = rewards
    
    # Smooth the rewards for better visualization (optional)
    def smooth_rewards(rewards, window_size=10):
        if len(rewards) < window_size:
            return rewards  # Return original if not enough data
        return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Plot Q-learning results
    plt.figure(figsize=(12, 8))
    for (alpha, epsilon), rewards in q_learning_results.items():
        smoothed_rewards = smooth_rewards(rewards)
        plt.plot(smoothed_rewards, label=f'α={alpha}, ε={epsilon}')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('Q-Learning Performance with Different Hyperparameters')
    plt.legend()
    plt.grid(True)
    plt.savefig('q_learning_windy_cliff_hyperparameters.png')
    print("Saved Q-learning hyperparameter plot")
    
    # Plot SARSA results
    plt.figure(figsize=(12, 8))
    for (alpha, epsilon), rewards in sarsa_results.items():
        smoothed_rewards = smooth_rewards(rewards)
        plt.plot(smoothed_rewards, label=f'α={alpha}, ε={epsilon}')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('SARSA Performance with Different Hyperparameters')
    plt.legend()
    plt.grid(True)
    plt.savefig('sarsa_windy_cliff_hyperparameters.png')
    print("Saved SARSA hyperparameter plot")

# Run the experiments
if __name__ == "__main__":
    print("Starting Windy Cliff World experiments...")
    
    # Testing Q-Learning
    print("Training Q-learning agent...")
    env = WindyCliffWorld()
    q_table, _ = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    visualize_policy(env, q_table, filename='q_learning_windy_cliff.gif')

    # Testing SARSA
    print("Training SARSA agent...")
    env = WindyCliffWorld()
    q_table, _ = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    visualize_policy(env, q_table, filename='sarsa_windy_cliff.gif')
    
    # Run hyperparameter experiments
    run_hyperparameter_experiments()
    
    print("All experiments completed!")