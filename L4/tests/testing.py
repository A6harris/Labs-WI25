# First add the correct imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from bandits import update, greedy, egreedy, empirical_egreedy
import numpy as np  # Add numpy import

# Test 1: update function
print("Testing update function...")
q = 0.5
r = 1.0
k = 1
result = update(q, r, k)
print(f"Update test (q={q}, r={r}, k={k}): {result}")

# Test 2: greedy function
print("\nTesting greedy function...")
q_estimate = np.array([0.1, 0.3, 0.2])
action = greedy(q_estimate)
print(f"Greedy test (q_values={q_estimate}): {action}")

# Test 3: egreedy function
print("\nTesting egreedy function...")
epsilon = 0.5
action = egreedy(q_estimate, epsilon)
print(f"E-greedy test (epsilon={epsilon}): {action}")

# Test 4: empirical_egreedy with small values
print("\nTesting empirical_egreedy function...")
test_rewards = empirical_egreedy(
    epsilon=0.1,
    n_trials=2,
    n_arms=3,
    n_plays=4
)

print(f"Number of trials: {len(test_rewards)}")
if len(test_rewards) > 0:
    print(f"Number of plays in first trial: {len(test_rewards[0])}")
    print(f"First trial rewards: {test_rewards[0]}")
    print(f"Second trial rewards: {test_rewards[1]}")