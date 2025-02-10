import numpy as np
import pandas as pd
from typing import List


def update(q: float, r: float, k: int) -> float:
    """
    Update the Q-value using the given reward and number of times the action has been taken.

    Parameters:
    q (float): The current Q-value.
    r (float): The reward received for the action.
    k (int): The number of times the action has been taken before.

    Returns:
    float: The updated Q-value.
    """
    # Note: since k is the number of times the action has been taken before this update, we need to add 1 to k before using it in the formula.
    # TODO
    a = 1/ (k+1)
    return q + a * (r - q)


def greedy(q_estimate: np.ndarray) -> int:
    """
    Selects the action with the highest Q-value.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of Q-values for each action.

    Returns:
    int: The index of the action with the highest Q-value.
    """
    # TODO
    return np.argmax(q_estimate)

def egreedy(q_estimate: np.ndarray, epsilon: float) -> int:
    """
    Implements the epsilon-greedy exploration strategy for multi-armed bandits.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of estimated action values.
    epsilon (float): Exploration rate, determines the probability of selecting a random action.
    n_arms (int): Number of arms in the bandit. default is 10.

    Returns:
    int: The index of the selected action.
    """
    # TODO
    n_arms = len(q_estimate)
    if np.random.random() < epsilon:
        greedy_action = np.argmax(q_estimate)
        explore = [a for a in range(n_arms) if a != greedy_action]
        return np.random.choice(explore)
    else:
        return np.argmax(q_estimate)


def empirical_egreedy(epsilon: float, n_trials: int, n_arms: int, n_plays: int) -> List[List[float]]:
    """
    Run epsilon-greedy algorithm on multi-armed bandit problem.
    
    Args:
        epsilon (float): Probability of exploring
        n_trials (int): Number of independent trials
        n_arms (int): Number of arms (actions)
        n_plays (int): Number of plays per trial
    
    Returns:
        List[List[float]]: List of rewards for each play in each trial
    """
    rewards = []  
    

    for i in range(n_trials):
        q_estimates = np.zeros(n_arms)
        counts = np.zeros(n_arms)
        trial_rewards = []

        true_rewards = np.random.normal(0, 1, size=n_arms)

        for p in range(n_plays):
            action = egreedy(q_estimates, epsilon)
            reward = np.random.normal(true_rewards[action], 1)
            counts[action] += 1
            q_estimates[action] = update(q_estimates[action], reward, counts[action] - 1)
            trial_rewards.append(reward)
        print(f"Trial {i} length: {len(trial_rewards)}")
        rewards.append(trial_rewards)

    
    return rewards
