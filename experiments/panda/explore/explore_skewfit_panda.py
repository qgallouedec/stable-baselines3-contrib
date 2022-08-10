import os

import gym
import numpy as np
import panda_gym
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from toolbox.panda_utils import compute_coverage

from sb3_contrib import SkewFit

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    model = SkewFit(
        env,
        nb_models=50,
        power=-0.2,
        num_presampled_goals=128,
        distance_threshold=0.2,
        action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(env.action_space.shape[0]), np.ones(env.action_space.shape[0])),
        verbose=1,
    )
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/skewfit_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
