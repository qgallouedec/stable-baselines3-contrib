import os

import gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
from toolbox.panda_utils import cumulative_object_coverage

from sb3_contrib import GoExplore
from sb3_contrib.go_explore.cells import Downscale

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    cell_factory = Downscale(np.log(2.0) / np.log(10))
    model = GoExplore(SAC, env, cell_factory, verbose=1)
    model.explore(NUM_TIMESTEPS)
    buffer = model.archive
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/go_explore_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
