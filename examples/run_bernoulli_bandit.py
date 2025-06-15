from bayesmab import BayesianBandit
import numpy as np

# Create a bandit with 3 variants
true_rates = {"A": 0.10, "B": 0.12, "C": 0.08}
bandit = BayesianBandit(arms=['A', 'B', 'C'],true_rates=true_rates)

# Online sampling: pick the best arm and update with reward
for _ in range(10000):
    arm, _ = bandit.sample()
    reward = np.random.binomial(1, true_rates[arm])  # Simulated true rates
    bandit.update(arm, reward)

# Print current posterior parameters
print("Posteriors:", bandit.get_posteriors())

# Print estimated probability each arm is the best
print("Probability best:", bandit.get_prob_best())
bandit.plot_posteriors()
bandit.plot_posterior_means()
bandit.plot_regret()
bandit.plot_traffic_allocation()
