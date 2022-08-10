import os

import gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG
from toolbox.panda_utils import compute_coverage

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/ddpg_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
