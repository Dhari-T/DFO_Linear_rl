import copy
import numpy as np
import gymnasium as gym

from src.common import *


# Random policy
def random_policy_baseline(env_name, seed, n=TEST_EPISODES):
    env = gym.make(env_name)
    env.action_space.seed(seed)
    returns = []
    for ep in range(n):
        obs, info = env.reset(seed=seed * 1000 + ep)
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# Rule-based controller
def rule_weights(env_name, flipped=False):
    # a rule written by hand turned into policy weights and used on the raw unscaled observations

    if flipped:
        sign = -1
    else:
        sign = 1

    if env_name == "CartPole-v1":            # push the way the pole is falling
        weights = np.zeros(8)
        weights[[5, 7]] = sign
        weights[[4, 6]] = -sign
    else:                                    # Acrobot: torque with joint-2 velocity
        weights = np.zeros(18)
        weights[17] = sign
        weights[15] = -sign
    return weights


def rule_based_baseline(env_name, seed, flipped=False):
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    empty = RunningNormaliser(env.observation_space.shape[0])   # n=0 -> raw observations
    result = test_policy(rule_weights(env_name, flipped), env, empty, rng)
    env.close()
    return result


# Random search
def random_search(env_name, seed, budget=BUDGET, eps_per_eval=EPS_PER_EVAL, lower=-1.0, upper=1.0,
                  models_dir=MODELS_DIR):
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_weights = obs_dim * n_actions
    normaliser = RunningNormaliser(obs_dim)          # one shared normaliser per run

    best_weights = None
    best_score = -np.inf
    best_normaliser = None
    total_steps = 0
    solved_at = None
    curve = []
    while total_steps < budget:
        weights = rng.uniform(lower, upper, size=n_weights)
        score, steps = evaluate(weights, env, normaliser, eps_per_eval, rng)
        total_steps += steps
        if score > best_score:
            best_weights = weights
            best_score = score
            best_normaliser = copy.deepcopy(normaliser)
        if solved_at is None and best_score >= THRESHOLDS[env_name]:
            solved_at = total_steps
        curve.append({"method": "random_search", "env": env_name, "seed": seed,
                      "steps": total_steps, "best_score": best_score})

    test_mean, test_std = test_policy(best_weights, env, best_normaliser, rng)
    save_model(f"random_search_{env_name}_seed{seed}", best_weights, best_normaliser, models_dir)
    env.close()
    summary = {"method": "random_search", "env": env_name, "seed": seed,
               "test_mean": test_mean, "test_std": test_std,
               "steps_to_threshold": solved_at, "solved": test_mean >= THRESHOLDS[env_name],
               "steps_used": total_steps}
    return summary, curve
