import os

import gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
from toolbox.panda_utils import cumulative_object_coverage

from sb3_contrib import ICM

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    icm = ICM(
        scaling_factor=0.01,
        actor_loss_coef=0.001,
        inverse_loss_coef=0.001,
        forward_loss_coef=100,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    model = SAC("MlpPolicy", env, surgeon=icm, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/icm_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
