import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set seed
np.random.seed(500)

class SimpleMDP:
    def __init__(self, n_states, k_rewards, gamma=0.9, exponent=1.5):
        self.n_states = n_states
        self.k_rewards = k_rewards
        self.gamma = gamma  # Discount factor
        self.exponent = exponent  # Exponent for power-law distribution
        self.states = np.arange(n_states)
        self.value_function = np.zeros(n_states)
        self.rewards = np.zeros(n_states)
        self.transitions = self.generate_transitions()
        self.assign_rewards()

    def generate_transitions(self):
        transitions = {}
        for state in self.states:
            # Number of connections follows a power-law distribution
            num_connections = np.random.pareto(a=self.exponent) + 1
            num_connections = min(num_connections, self.n_states)
            connected_states = np.random.choice(self.states, int(num_connections), replace=False)
            transitions[state] = connected_states
        return transitions

    def assign_rewards(self):
        reward_states = np.random.choice(self.states, self.k_rewards, replace=False)
        for state in reward_states:
            self.rewards[state] = np.random.uniform(1, 10)  # Assign a random reward between 1 and 10

    def update_value(self, state):
        # Bellman update
        max_value = max([self.value_function[next_state] for next_state in self.transitions[state]])
        self.value_function[state] = self.rewards[state] + self.gamma * max_value

    def value_iteration(self, iterations=1000):
        value_functions = list()
        for _ in range(iterations):
            state = np.random.choice(self.states)
            self.update_value(state)
            value_functions.append(self.value_function.copy())

        return value_functions
    
def value_degree_correlation(mdp, value_function):
    # Compute degree of each state
    degrees = np.zeros(mdp.n_states)
    for state in mdp.states:
        degrees[state] = len(mdp.transitions[state])

    if np.std(degrees) == 0 or np.std(value_function) == 0:
        return 0

    # Compute correlation between degree and value function
    correlation = np.corrcoef(degrees, value_function)[0, 1]
    return correlation

def experiment(n_states, k_rewards, exponent, trials=10, iterations=6000):
    print(f"Running experiment with {n_states} states, {k_rewards} rewards, and exponent {exponent}")
    for i in tqdm(range(trials)):
        mdp = SimpleMDP(n_states=n_states, k_rewards=k_rewards, exponent = exponent)
        value_functions = mdp.value_iteration(iterations = iterations)
        correlations = list()
        for i, value_function in enumerate(value_functions):
            sorted_value_function = sorted(value_function)
            correlation = value_degree_correlation(mdp, value_function)
            correlations.append(correlation)

        plt.plot(correlations)
    plt.title(f'Iterations vs. Value-Degree Correlation')
    plt.xlabel('Iterations')
    plt.ylabel('Correlation')
    plt.savefig(f'iterations_vs_value_degree_correlation_states_{n_states}_rewards_{k_rewards}_exponent_{exponent}.png')
    plt.close()

experiment(n_states=100, k_rewards=1, exponent=1.5, trials=10)
experiment(n_states=100, k_rewards=1, exponent=5, trials=10)
experiment(n_states=1000, k_rewards=1, exponent=1.5, trials=10)
experiment(n_states=100, k_rewards=10, exponent=1.5, trials=10)