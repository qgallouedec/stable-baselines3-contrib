import os

import gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from toolbox.panda_utils import compute_coverage

from sb3_contrib import Surprise

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    surprise = Surprise(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        eta_0=0.1,
        feature_dim=16,
        lr=0.001,
        train_freq=4,
    )
    model = DDPG(
        "MlpPolicy",
        env,
        surgeon=surprise,
        action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(env.action_space.shape[0]), np.ones(env.action_space.shape[0])),
        verbose=1,
    )
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/surprise_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
