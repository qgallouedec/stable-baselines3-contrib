import copy
from typing import Any, Callable, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from sb3_contrib.go_explore.archive import ArchiveBuffer
from sb3_contrib.go_explore.feature_extractor import GoExploreExtractor


class VecGoalify(VecEnvWrapper):
    """
    Wrap the env into a GoalEnv.

    :param venv: The vectorized environment
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    :param distance_threshold: The goal is reached when the latent distance between
    the current obs and the goal obs is below this threshold, defaults to 1.0
    :param lighten_dist_coef: Remove subgoal that are not further than lighten_dist_coef*dist_threshold
        from the previous subgoal, defaults to 1.0
    """

    def __init__(self, venv: VecEnv, nb_random_exploration_steps: int = 30, window_size: int = 10) -> None:
        super().__init__(venv)
        cell_space = venv.get_attr("cell_space")[0]
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(venv.observation_space),
                "goal": copy.deepcopy(cell_space),
                "cell": copy.deepcopy(cell_space),
            }
        )
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size
        self.goal_trajectories = [None for _ in range(self.num_envs)]
        self.archive_buffer: ArchiveBuffer

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        assert hasattr(self, "archive_buffer"), "you need to set the buffer before reset. Use set_archive()"
        for env_idx in range(self.num_envs):
            goal_trajectory = self.archive_buffer.sample_trajectory()
            self.goal_trajectories[env_idx] = goal_trajectory

        self._goal_idxs = np.zeros(self.num_envs, dtype=np.int64)
        self.done_countdowns = self.nb_random_exploration_steps * np.ones(self.num_envs, dtype=np.int64)
        self._is_last_goal_reached = np.zeros(self.num_envs, dtype=bool)  # useful flag
        dict_observations = {
            "observation": observations,
            "goal": np.stack([self.goal_trajectories[env_idx][self._goal_idxs[env_idx]] for env_idx in range(self.num_envs)]),
            "cell": np.zeros(self.observation_space["cell"].shape, dtype=np.int64),
        }
        return dict_observations

    def set_archive(self, archive_buffer: ArchiveBuffer) -> None:
        """
        Set the buffer.

        The buffer is used to compute goal trajectories.

        :param buffer: The buffer
        """
        self.archive_buffer = archive_buffer

    def _get_dict_obs(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        return

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        return super().step_async(actions)

    def _reset_one_env(self, env_idx):
        if isinstance(self.venv, SubprocVecEnv):
            self.venv.remotes[env_idx].send(("reset", None))
            return self.venv.remotes[env_idx].recv()
        if isinstance(self.venv, DummyVecEnv):
            return self.venv.envs[env_idx].reset()

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        for info, reward in zip(infos, rewards):
            info["env_reward"] = reward

        cells = np.array([info["cell"] for info in infos])
        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        for env_idx in range(self.num_envs):
            infos[env_idx]["is_success"] = self._is_last_goal_reached[env_idx]  # Will be overwritten if necessary
            if not dones[env_idx]:
                upper_idx = min(self._goal_idxs[env_idx] + self.window_size, len(self.goal_trajectories[env_idx]))
                future_goals = self.goal_trajectories[env_idx][self._goal_idxs[env_idx] : upper_idx]
                flat_future_goals = future_goals.reshape(future_goals.shape[0], -1)
                flat_cell = cells[env_idx].flatten()
                future_success = (flat_cell == flat_future_goals).all(-1)
                if future_success.any():
                    furthest_futur_success = np.where(future_success)[0].max()
                    self._goal_idxs[env_idx] += furthest_futur_success + 1
                if self._goal_idxs[env_idx] == len(self.goal_trajectories[env_idx]):
                    self._is_last_goal_reached[env_idx] = True
                    self._goal_idxs[env_idx] -= 1

                # When the last goal is reached, delay the done to allow some random actions
                if self._is_last_goal_reached[env_idx]:
                    infos[env_idx]["is_success"] = True
                    if self.done_countdowns[env_idx] != 0:
                        infos[env_idx]["action_repeat"] = self.actions[env_idx]
                        self.done_countdowns[env_idx] -= 1
                    else:  # self.done_countdown == 0:
                        dones[env_idx] = True
                        terminal_observation = observations[env_idx]
                        observations[env_idx] = self._reset_one_env(env_idx)
                        cells[env_idx] = np.zeros(self.observation_space["cell"].shape, dtype=np.int64)

            # Dones can be due to env (death), or to the previous code
            if dones[env_idx]:
                # If done is due to inner env, terminal obs is already in infos. Else
                # it is written in terminal obs, see above.
                if "terminal_observation" in infos[env_idx]:
                    terminal_observation = infos[env_idx]["terminal_observation"]
                infos[env_idx]["terminal_observation"] = {
                    "observation": terminal_observation,
                    "goal": self.goal_trajectories[env_idx][self._goal_idxs[env_idx]],
                    "cell": cells[env_idx],
                }
                self.goal_trajectories[env_idx] = self.archive_buffer.sample_trajectory()
                self._goal_idxs[env_idx] = 0
                self.done_countdowns[env_idx] = self.nb_random_exploration_steps
                self._is_last_goal_reached[env_idx] = False

        dict_observations = {
            "observation": observations,
            "goal": np.stack([self.goal_trajectories[env_idx][self._goal_idxs[env_idx]] for env_idx in range(self.num_envs)]),
            "cell": cells,
        }
        return dict_observations, rewards, dones, infos


class CallEveryNTimesteps(BaseCallback):
    """
    Callback that calls a function every ``call_freq`` timesteps.

    :param func: The function to call
    :param call_freq: The call timestep frequency, defaults to 1
    :param verbose: Verbosity level 0: not output 1: info 2: debug, defaults to 0
    """

    def __init__(self, func: Callable[[], None], call_freq: int = 1, verbose=0) -> None:
        super(CallEveryNTimesteps, self).__init__(verbose)
        self.func = func
        self.call_freq = call_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps // self.model.n_envs % (self.call_freq // self.model.n_envs) == 0:
            self.func()

        return True


class GoExplore:
    """
    This is a simplified version of Go-Explore from the original paper, which does not include
    a number of tricks which impacts the performance.
    The goal is to implement the general principle of the algorithm and not all the little tricks.
    In particular, we do not implement:
    - everything related with domain knowledge,
    - self-imitation learning,
    - parallelized exploration phase
    """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env_id: str,
        n_envs: int = 1,
        env_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        learning_starts: int = 100,
        model_kwargs: Optional[Dict[str, Any]] = None,
        wrapper_cls: Optional[gym.Wrapper] = None,
        nb_random_exploration_steps: int = 50,
        vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        self.device = get_device(device)

        env_kwargs = {} if env_kwargs is None else env_kwargs
        venv = make_vec_env(env_id, n_envs=n_envs, wrapper_class=wrapper_cls, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)
        venv = VecGoalify(venv, nb_random_exploration_steps=nb_random_exploration_steps)
        venv = VecMonitor(venv)

        cell_dim = np.sum(venv.get_attr("cell_space")[0].nvec)  # size of multidiscrete

        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_kwargs["learning_starts"] = learning_starts
        model_kwargs["train_freq"] = 10
        model_kwargs["gradient_steps"] = n_envs
        model_kwargs["policy_kwargs"]["features_extractor_class"] = GoExploreExtractor
        model_kwargs["policy_kwargs"]["features_extractor_kwargs"] = dict(cell_dim=cell_dim)
        self.model = model_class(
            "MultiInputPolicy",
            venv,
            replay_buffer_class=ArchiveBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **model_kwargs,
        )
        self.archive = self.model.replay_buffer  # type: ArchiveBuffer
        venv.set_archive(self.archive)

    def explore(self, total_timesteps: int, update_cell_factory_freq=1_000, callback: MaybeCallback = None) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration
        :param update_freq: Cells update frequency
        """
        cell_factory_updater = CallEveryNTimesteps(self.archive.recompute_weights, update_cell_factory_freq)
        if callback is not None:
            callback = [cell_factory_updater, callback]
        else:
            callback = [cell_factory_updater]
        self.model.learn(total_timesteps, callback=callback)
