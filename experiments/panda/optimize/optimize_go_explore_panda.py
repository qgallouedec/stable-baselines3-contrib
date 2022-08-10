import gym
import numpy as np
import optuna
import panda_gym
from stable_baselines3 import DDPG
from toolbox.panda_utils import compute_coverage

from sb3_contrib import GoExplore
from sb3_contrib.go_explore.cells import Downscale

NUM_TIMESTEPS = 300_000
NUM_RUN = 3


def objective(trial: optuna.Trial) -> float:
    decimals = trial.suggest_categorical("decimals", [0.0, 0.5, 1.0, 1.5, 2.0])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        cell_factory = Downscale(decimals)
        model = GoExplore(
            DDPG,
            env,
            cell_factory,
            verbose=1,
        )
        model.explore(NUM_TIMESTEPS)
        buffer = model.archive
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = compute_coverage(observations)

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    from optuna.samplers import GridSampler

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="go_explore_panda",
        load_if_exists=True,
        direction="maximize",
        sampler=GridSampler(dict(decimals=[0.0, 0.5, 1.0, 1.5, 2.0])),
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
