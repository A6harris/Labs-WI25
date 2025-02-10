import numpy as np

import matplotlib.pyplot as plt
from src.maze_env import MazeEnv
from typing import List, Callable, Tuple
import matplotlib.animation as animation
import random

# Define the states and possible actions
states = np.arange(1, 6)  # States 1 through 5
actions = ['left', 'stay', 'right']  # Available actions in each state


def transition(state, action):
    """
    Transition function that determines the next state based on the current state and action.

    Parameters:
    state (int): The current state.
    action (str): The action chosen.

    Returns:
    int: The next state.
    """
    # TODO: Your code here
    if action == 'stay':
        return state
    elif action == 'left':
        if state > 1 and state <= 6:
            state -= 1
            return state
    elif action == 'right':
        if state >= 1 and state < 6:
            state += 1
            return state
    return (state)

def reward(state, action):
    """
    Calculate the reward for a given state and action.

    Parameters:
    state (int): The current state.
    action (str): The action taken.

    Returns:
    int: The reward.
    """
    # TODO: Your code here
    if state == 4 and action == 'right':
        reward = 10
        return reward
    else:
        reward = -1
        return reward




def always_right_policy(state):
    """
    Policy that always returns 'right' for any given state.

    Parameters:
    state (int): The current state.

    Returns:
    str: The chosen action ('right').
    """
    return 'right'


def my_policy(state):
    """
    This function implements a custom policy.

    Parameters:
    state (int): The current state of the system.

    Returns:
    str: The action chosen by the policy.
    """
    # TODO: Your code here
    i = random.random()
    if state >= 1 and state < 3:
        if i < 0.5:
            return 'right'
        elif i >= 0.5:
            return 'stay'
    if state < 3 and state <= 6:
        if i < 0.5:
            return 'left'
        elif i >= 0.5:
            return 'stay'
    elif i < 0.3:
        return 'left'
    elif i >= 0.3 and i < 0.6:
        return 'stay'
    elif i >= 0.6:
        return 'right'

        

        
def simulate_mdp(policy: Callable, initial_state=1, simulation_depth=20):
    """
    Simulates the Markov Decision Process (MDP) based on the given policy. 
    If we reach the terminal state, the simulation ends.
    Keeps track of the number of visits to each state, the cumulative reward, and the history of visited states.

    Parameters:
    - policy: A function that takes the current state as input and returns an action.
    - initial_state: The initial state of the MDP. Default is 1.
    - simulation_depth: The maximum number of steps to simulate. Default is 20.

    Returns:
    - state_visits: An array that tracks the number of visits to each state.
    - cumulative_reward: The cumulative reward obtained during the simulation.
    - visited_history: A list that tracks the history of visited states.
    - reward_history: A list that tracks the history of rewards obtained.
    """
    current_state = initial_state
    cumulative_reward = 0
    state_visits = np.zeros(len(states))
    visited_history = [current_state] 
    reward_history = [0]
    
    for _ in range(simulation_depth):
        action = policy(current_state)
        current_reward = reward(current_state, action)
        cumulative_reward += current_reward
        state_visits[current_state - 1] += 1
        next_state = transition(current_state, action)
        reward_history.append(current_reward)
        visited_history.append(next_state)
        if current_state == 4 and action == 'right':
            break
            
        current_state = next_state
    
    return state_visits, cumulative_reward, visited_history, reward_history

def new_policy(state: List[int]) -> int:
    current_state = state[0] if isinstance(state, (list, tuple)) else state

    if current_state == 4:
        return 0 
    elif current_state < 4:
        return 0  
    elif current_state > 4:
        return 1  
    return 2

        
def simulate_maze_env(env: MazeEnv, policy: Callable, num_steps=20):
    """
    Simulates the environment using the given policy for a specified number of steps.

    Parameters:
    - env: The environment to simulate.
    - policy: The policy to use for selecting actions (this is a function that takes a state as input and returns an action)
    - num_steps: The number of steps to simulate (default: 20).

    Returns:
    - path: The sequence of states visited during the simulation.
    - total_reward: The total reward accumulated during the simulation.
    """
    state = env.reset()
    total_reward = 0
    path = [state]

    for _ in range(num_steps):
        # TODO: Your code here
        action = policy(state)
        
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        path.append(next_state)
        
        state = next_state
        if done:
            break
    return path, total_reward


def q_learning(env: MazeEnv, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1) -> np.ndarray:
    """
    Perform Q-learning to learn the optimal policy for the given environment.

    Args:
        env (MazeEnv): The environment to learn the policy for.
        episodes (int, optional): Number of episodes for training. Defaults to 500.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Exploration rate. Defaults to 0.1.

    Returns:
        np.ndarray: The learned Q-table.
    """
    """
    Perform Q-learning to learn the optimal policy for the given environment.
    Modified to handle both Discrete and Box observation spaces with proper state dimensioning.

    Args:
        env (MazeEnv): The environment to learn the policy for.
        episodes (int, optional): Number of episodes for training. Defaults to 500.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Exploration rate. Defaults to 0.1.

    Returns:
        np.ndarray: The learned Q-table.
    """
    states_seen = set()
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    states_seen.add(int(state[0]) if isinstance(state, (tuple, list, np.ndarray)) else int(state))
    
    # Sample some random actions to explore state space
    for _ in range(100):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)
        state_val = int(next_state[0]) if isinstance(next_state, (tuple, list, np.ndarray)) else int(next_state)
        states_seen.add(state_val)
        if done:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            states_seen.add(int(state[0]) if isinstance(state, (tuple, list, np.ndarray)) else int(state))
    
    # Calculate grid size
    state_size = int(np.sqrt(max(states_seen) + 1))
    
    # Initialize Q-table
    q_table = np.zeros((state_size, state_size, env.action_space.n))
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        
        while not done:
            # Get current state index
            state_val = int(state[0]) if isinstance(state, (tuple, list, np.ndarray)) else int(state)
            row = state_val // state_size
            col = state_val % state_size
            
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[row, col])
            
            # Take action
            next_state, reward, done, info = env.step(action)
            if isinstance(info, tuple):
                done = info[0]
                info = info[1]
            
            # Get next state indices
            next_state_val = int(next_state[0]) if isinstance(next_state, (tuple, list, np.ndarray)) else int(next_state)
            next_row = next_state_val // state_size
            next_col = next_state_val % state_size
            
            # Update Q-value
            old_value = q_table[row, col, action]
            next_max = np.max(q_table[next_row, next_col])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[row, col, action] = new_value
            
            state = next_state
            
            if done:
                break
    
    return q_table
    
    return q_table
def simulate_maze_env_q_learning(
    env: MazeEnv, q_table: np.ndarray
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate the maze environment using the Q-table to determine the actions to take.
    Also creates an animation of the agent moving through the environment.
    
    Args:
        env (MazeEnv): The maze environment instance.
        q_table (np.ndarray): The Q-table containing action values.

    Returns:
        Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of states and a boolean indicating if the episode is done.
    """

    state = env.reset()
    done = False

    starting_frame = env.render(mode="rgb_array").T
    frames = [starting_frame]  # List to store frames for animation
    states = [state]  # List to store states

    while not done:
        state_idx = int(state[0]) if isinstance(state, (tuple, list, np.ndarray)) else int(state)
        action = np.argmax(q_table[state_idx]) # TODO: Your code here
        state, _, done, _ = env.step(action)
        frames.append(
            env.render(mode="rgb_array").T
        )  # Render the environment as an RGB array
        states.append(state)

    def update_frame(i):
        ax.clear()
        ax.imshow(frames[i], cmap="viridis", origin="lower")
        ax.set_title(f"Step {i+1}")
        ax.grid("on")

    # Create animation from frames
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=500)
    anim.save("mdp_q_learning.gif", writer="pillow")
    return states, done