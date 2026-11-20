# stuff the other files share like folder paths settings the normaliser the linear policy and evaluation
import os
import numpy as np

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
for folder in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    os.makedirs(folder, exist_ok=True)

# Experiment settings
ENVS = ["CartPole-v1", "Acrobot-v1"]
THRESHOLDS = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0}
BUDGET = 500_000          # environment steps per run (matches Wong et al. 2024)
EPS_PER_EVAL = 3          # episodes per fitness evaluation (same for all optimisers)
TEST_EPISODES = 20        # fresh episodes for the final test
SEEDS = list(range(21))  # testing, use list(range(21)) for the full run for 20 seeds or list(range(6)) for 5 seeds


# Observation normaliser (running mean / variance)
class RunningNormaliser:
    def __init__(self, obs_dim):
        self.n = 0
        self.mean = np.zeros(obs_dim)
        self.m2 = np.zeros(obs_dim)

    def update(self, observation):
        self.n += 1
        delta = observation - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (observation - self.mean)      # uses the NEW mean

    def normalise(self, observation):
        if self.n == 0:
            return observation                            # no statistics yet -> leave unchanged (avoids 0/0 = nan)
        variance = self.m2 / self.n
        return (observation - self.mean) / np.sqrt(variance + 1e-8)


# Linear policy
def linear_policy(weights, obs_norm, obs_dim, n_actions):
    weight_matrix = weights.reshape(obs_dim, n_actions)     # e.g. 18 numbers -> 6x3 matrix (one column per action)
    scores = obs_norm @ weight_matrix                       # one score per action
    return int(np.argmax(scores))


# Evaluate a policy
def evaluate(weights, env, normaliser, n_episodes, rng, update_norm=True):
    # plays n_episodes games with these weights and returns the average score and the total steps used
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    episode_returns = []
    total_steps = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1e9)))
        ep_return = 0.0
        done = False

        while not done:
            if update_norm:
                normaliser.update(obs)            # update stats first...
            obs_norm = normaliser.normalise(obs)  # ...then scale the observation
            action = linear_policy(weights, obs_norm, obs_dim, n_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            total_steps += 1
            done = terminated or truncated

        episode_returns.append(ep_return)

    return float(np.mean(episode_returns)), total_steps


# Final test and model saving
def test_policy(weights, env, normaliser, rng, n=TEST_EPISODES):
    # final test that plays n brand new games with the normaliser frozen and returns the average score and its spread
    returns = []
    for episode in range(n):
        score, steps = evaluate(weights, env, normaliser, 1, rng, update_norm=False)
        returns.append(score)
    return float(np.mean(returns)), float(np.std(returns))


def save_model(name, weights, normaliser, models_dir=MODELS_DIR):
    # saves the weights together with the normaliser stats because the policy needs both to work again later
    os.makedirs(models_dir, exist_ok=True)
    var = normaliser.m2 / max(normaliser.n, 1)
    np.savez(os.path.join(models_dir, f"{name}.npz"),
             weights=weights, mean=normaliser.mean, var=var, n=normaliser.n)
