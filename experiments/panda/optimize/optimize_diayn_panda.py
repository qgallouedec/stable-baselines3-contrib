import gym
import numpy as np
import optuna
import panda_gym
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from toolbox.panda_utils import compute_coverage

from sb3_contrib import DIAYN

NUM_TIMESTEPS = 300_000
NUM_RUN = 3


def objective(trial: optuna.Trial) -> float:
    nb_skills = trial.suggest_categorical("nb_skills", [4, 8, 16, 32, 64, 128])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        model = DIAYN(
            env,
            nb_skills,
            action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(env.action_space.shape[0]), np.ones(env.action_space.shape[0])),
            verbose=1,
        )
        model.learn(NUM_TIMESTEPS)
        buffer = model.replay_buffer
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = compute_coverage(observations)

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db", study_name="diayn_panda", load_if_exists=True, direction="maximize"
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
