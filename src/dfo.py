import copy
import numpy as np
import gymnasium as gym

from src.common import *

# dfo settings
N_FLIES = 10       # number of flies
DT = 0.001         # disturbance threshold
DT_MAX = 0.1       # highest dt that adaptive dt can reach
PATIENCE = 3       # rounds without improvement before adaptive dt kicks in
LOWER = -1.0       # lowest allowed weight
UPPER = 1.0        # highest allowed weight

VARIANTS = {"dfo":                 dict(),
            "dfo_adaptive":        dict(adaptive_dt=True),
            "dfo_reeval":          dict(reeval_eps=10),
            "dfo_adaptive_reeval": dict(adaptive_dt=True, reeval_eps=10)}


def dfo(env_name, seed, method_name="dfo", adaptive_dt=False, reeval_eps=0, n_flies=N_FLIES):

    """ dispersive flies optimisation where a swarm of flies searches for good policy weights
    # each fly gets the average score of its games so higher is better and it stops once the step budget is used up
    # adaptive_dt makes the flies scatter more by multiplying dt by 10 up to DT_MAX when the best score stops improving for PATIENCE rounds and resets once it improves
    # reeval_eps makes every new best play that many extra games and only keeps it if the combined average still beats the old best which guards against lucky scores
    # returns the summary row and the learning curve rows """

    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_dims = obs_dim * n_actions
    normaliser = RunningNormaliser(obs_dim)          # one shared normaliser per run

    flies = rng.uniform(LOWER, UPPER, size=(n_flies, n_dims))             # each row is one fly = one weight vector
    scores = np.zeros(n_flies)

    best_weights = None
    best_score = -np.inf
    best_normaliser = None
    total_steps = 0
    solved_at = None
    curve = []
    cur_dt = DT
    stall = 0

    while total_steps < BUDGET:
        # evaluate every fly
        for fly in range(n_flies):
            score, steps = evaluate(flies[fly], env, normaliser, EPS_PER_EVAL, rng)
            scores[fly] = score
            total_steps += steps

        # swarm best and best-ever bookkeeping
        best_fly = int(np.argmax(scores))
        improved = False
        if scores[best_fly] > best_score:
            candidate = scores[best_fly]
            if reeval_eps > 0:
                extra, steps = evaluate(flies[best_fly], env, normaliser, reeval_eps, rng)
                total_steps += steps
                candidate = (candidate * EPS_PER_EVAL + extra * reeval_eps) / (EPS_PER_EVAL + reeval_eps)
            if candidate > best_score:
                best_score = candidate
                best_weights = flies[best_fly].copy()
                best_normaliser = copy.deepcopy(normaliser)    # the scaling these weights were scored with
                improved = True
        if solved_at is None and best_score >= THRESHOLDS[env_name]:
            solved_at = total_steps

        # adaptive disturbance threshold
        if adaptive_dt:
            if improved:
                stall = 0
            else:
                stall = stall + 1
            if stall < PATIENCE:
                cur_dt = DT
            else:
                cur_dt = min(DT_MAX, cur_dt * 10)

        curve.append({"method": method_name, "env": env_name, "seed": seed,
                      "steps": total_steps, "best_score": best_score, "dt": cur_dt})

        # update every fly except the swarm best (elitism)
        for fly in range(n_flies):
            if fly == best_fly:
                continue
            left = (fly - 1) % n_flies
            right = (fly + 1) % n_flies
            if scores[left] > scores[right]:
                neighbour = left
            else:
                neighbour = right
            for dim in range(n_dims):
                if rng.random() < cur_dt:
                    flies[fly, dim] = rng.uniform(LOWER, UPPER)                        # disturbance
                else:
                    random_factor = rng.random()
                    flies[fly, dim] = flies[neighbour, dim] + random_factor * (flies[best_fly, dim] - flies[fly, dim])
                if flies[fly, dim] < LOWER or flies[fly, dim] > UPPER:
                    flies[fly, dim] = rng.uniform(LOWER, UPPER)                        # out of bounds -> reset

    test_mean, test_std = test_policy(best_weights, env, best_normaliser, rng)
    save_model(f"{method_name}_{env_name}_seed{seed}", best_weights, best_normaliser, MODELS_DIR)
    env.close()

    summary = {"method": method_name, "env": env_name, "seed": seed,
               "test_mean": test_mean, "test_std": test_std,
               "steps_to_threshold": solved_at, "solved": test_mean >= THRESHOLDS[env_name],
               "steps_used": total_steps}
    return summary, curve
