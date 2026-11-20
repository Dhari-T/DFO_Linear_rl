# run this once after setting up the environment to check everything works and it only takes a few seconds
import sys
import platform
import numpy as np
import matplotlib
import gymnasium as gym

from src.common import *


def versions():
    print("Python    ", sys.version.split()[0])
    print("Platform  ", platform.platform())
    print("numpy     ", np.__version__)
    print("matplotlib", matplotlib.__version__)
    print("gymnasium ", gym.__version__)
    print("Project   ", PROJECT_DIR)


def env_check():
    print("\n--- Environments ---")
    for name in ["CartPole-v1", "Acrobot-v1"]:
        env = gym.make(name)
        env.reset(seed=0)
        print(f"{name}: observation dims = {env.observation_space.shape[0]}, "
              f"actions = {env.action_space.n}, solved threshold = {gym.spec(name).reward_threshold}")
        env.close()


def policy_tests():
    rng = np.random.default_rng(0)

    # Normaliser: normalised observations should have mean ~0, std ~1
    env = gym.make("Acrobot-v1")
    normaliser = RunningNormaliser(env.observation_space.shape[0])
    raw = []
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            normaliser.update(obs)
            raw.append(obs)
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
    normalised = np.array([normaliser.normalise(observation) for observation in raw])
    print("\nTest 1 - normaliser")
    print("mean per column:", np.round(normalised.mean(axis=0), 3))
    print("std  per column:", np.round(normalised.std(axis=0), 3))

    # Evaluate on Acrobot
    n_weights = env.observation_space.shape[0] * env.action_space.n          # 18
    print("\nTest 2 - evaluate on Acrobot")
    print("zero weights :", evaluate(np.zeros(n_weights), env, RunningNormaliser(6), 2, rng), "expect (-500.0, 1000)")
    acrobot_weights = np.zeros(n_weights)                                               # row 5 (elbow velocity): left -1, right +1
    acrobot_weights[15] = -1
    acrobot_weights[17] = 1
    print("rule weights :", evaluate(acrobot_weights, env, RunningNormaliser(6), 5, rng), " expect mean ~ -60 to -90")
    env.close()

    # Hand-set CartPole weights: push right when the pole falls right
    env = gym.make("CartPole-v1")
    cartpole_weights = np.zeros(8)
    cartpole_weights[[5, 7]] = 1
    cartpole_weights[[4, 6]] = -1
    print("\nTest 3 - CartPole hand set weights")
    print("3a raw :", evaluate(cartpole_weights, env, RunningNormaliser(4), 5, rng, update_norm=False),
          " expect mean ~400-500")
    print("3b normalised obs:", evaluate(cartpole_weights, env, RunningNormaliser(4), 5, rng), " expect lower / variable")
    env.close()


if __name__ == "__main__":
    versions()
    env_check()
    policy_tests()
    print("\nAll checks ran.")
