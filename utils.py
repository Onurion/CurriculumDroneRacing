import numpy as np
import os
import warnings
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Union, Optional, List, Callable
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import type_aliases

# ----------------------------------------
# SelfPlayWrapper: Present training agent's perspective (drone0)
# with frozen opponent policy for drone1.
# ----------------------------------------
class SelfPlayWrapper(gym.Env):
    def __init__(self, env, frozen_opponent_policies, return_dict=False):
        self.env = env
        # self.frozen_opponent_policy = frozen_opponent_policy
        self.frozen_opponent_policies = frozen_opponent_policies
        self.observation_space = env.observation_space["drone0"]
        self.action_space = env.action_space["drone0"]
        self.last_full_obs = None
        self.vehicles = env.vehicles
        # self.return_dict = return_dict

    def reset(self, **kwargs):
        full_obs, info = self.env.reset(**kwargs)
        self.last_full_obs = full_obs
        return full_obs["drone0"], info

    def step(self, action):
        actions = {"drone0": action}  # Main agent's action

        # print("Available opponent policies:", list(self.frozen_opponent_policies.keys()))

        # Get actions for all opponent agents
        for agent_id, policy in self.frozen_opponent_policies.items():
            opponent_obs = self.last_full_obs[agent_id]
            opponent_action = policy(opponent_obs)
            # print(f"Agent {agent_id} action:", opponent_action)
            actions[agent_id] = opponent_action

        full_obs, rewards, dones, truncated, info = self.env.step(actions)
        self.last_full_obs = full_obs
        return full_obs["drone0"], rewards["drone0"], dones["__all__"], truncated, info

    def render(self, mode='human'):
        self.env.render(mode)

    @property
    def agents(self):
        # Return the agents attribute from the inner environment.
        return self.env.agents

    def get_state(self, agent):
        return self.env.get_state(agent)


def evaluate_policies(env, training_model, num_episodes=1):
    wins = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action_tr, _ = training_model.predict(obs, deterministic=True)
            obs, rewards, dones, trunc, info = env.step(action_tr)
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = dones

        try:
            # Skip evaluation if the info structure isn't as expected
            if "drone0" not in info:
                print(f"Episode {episode}: Missing drone0 in info")
                continue

            # Get training drone's progress
            progress_tr = info["drone0"].get("progress", 0)

            # Check against all opponent drones
            win_current_episode = True
            for drone_id, drone_info in info.items():
                if (drone_id != "drone0" and
                    drone_id != "episode" and  # Skip the episode info
                    isinstance(drone_info, dict) and
                    "progress" in drone_info):

                    if progress_tr <= drone_info["progress"]:
                        win_current_episode = False
                        break

            if win_current_episode:
                wins += 1

        except Exception as e:
            print(f"Episode {episode}: Error processing info - {e}")
            print(f"Current info structure: {info}")
            continue

    return wins / num_episodes

# ----------------------------------------
# Frozen opponent policy using a frozen model.
# ----------------------------------------
class FrozenOpponentPolicy:
    def __init__(self, model):
        self.model = model  # This is a PPO model.
    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

# Custom callback that evaluates and updates the frozen opponent model.
class FrozenModelUpdateCallback(BaseCallback):
    def __init__(self, eval_env, frozen_model, eval_freq: int, eval_episodes: int, win_rate_threshold: float, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env  # A single-instance evaluation environment.
        self.frozen_model = frozen_model
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold

    def _on_step(self) -> bool:
        # Evaluate at the specified frequency.
        if self.n_calls % self.eval_freq == 0:
            win_rate = evaluate_policies(
                self.eval_env, self.model, num_episodes=self.eval_episodes
            )
            if self.verbose:
                print(f"\n[FrozenModelUpdateCallback] Evaluation win rate: {win_rate:.2f}")
            if win_rate >= self.win_rate_threshold:
                # Update the frozen model.
                self.frozen_model.set_parameters(self.model.get_parameters())
                if self.verbose:
                    print("--> Frozen model updated based on evaluation criteria.")
        return True



def write_env_parameters(main_folder, env_args, **kwargs):
    # Merge env_args with extra parameters from kwargs
    parameters = {**env_args, **kwargs}
    params_file = os.path.join(main_folder, "parameters.txt")
    with open(params_file, "a") as f:
        f.write("Training parameters:\n")
        for key, value in parameters.items():
            # Format floats to two decimal places
            if isinstance(value, float):
                # Attempt to write as formatted float, otherwise fallback to the default string representation
                try:
                    f.write(f"{key}= {value:.2f}\n")
                except Exception:
                    f.write(f"{key}= {value}\n")
            else:
                f.write(f"{key}= {value}\n")
        f.write("\n")


def log_trial_results(main_folder, trial_number, score, mean_targets_reached, std_targets_reached,
                      mean_speed, std_speed, mean_collisions, std_collisions):
    log_line = (f"Trial {trial_number} Score: {score:.2f}, "
                f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

    with open(os.path.join(main_folder, "all_trials.txt"), "a") as f:
        f.write(log_line)



def print_results(main_folder, stage, score, mean_targets_reached, std_targets_reached,
                      mean_speed, std_speed, mean_collisions, std_collisions):
    log_line = (f"Level: {stage} Score: {score:.2f}, "
                f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

    with open(os.path.join(main_folder, "results.txt"), "w") as f:
        f.write(log_line)


def evaluate_policy_updated(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[
    tuple[float, float, dict[str, float]],
    tuple[list[float], list[int], dict[str, list[float]]]
]:
    """
    Runs policy for ``n_eval_episodes`` and returns evaluation metrics along with
    extra info (custom metrics such as targets reached, speed, and collision).

    If a vectorized environment is passed in (even with n_envs > 1),
    the episodes are divided among the different workers as evenly as possible.
    This helps remove bias in the evaluation, as discussed in:
    https://github.com/DLR-RM/stable-baselines3/issues/402

    .. note::
        If the environment is not wrapped with a ``Monitor`` wrapper, the rewards and
        episode lengths are computed directly from env.step outputs. If the environment
        includes wrappers that modify rewards, you might see differences.

    :param model: The RL agent to evaluate (must implement a `predict` method).
    :param env: The gym environment or VecEnv.
    :param n_eval_episodes: Number of episodes to evaluate.
    :param deterministic: Whether to use deterministic actions.
    :param render: Whether to render the environment during evaluation.
    :param callback: A callback function called after each step (receives locals() and globals()).
    :param reward_threshold: A minimum expected reward per episode; an error is raised if not met.
    :param return_episode_rewards:
        If True, returns per-episode rewards, lengths, and raw custom info lists;
        otherwise returns aggregated (mean, std) rewards and mean custom info.
    :param warn: Whether to warn the user if the evaluation environment is not wrapped with Monitor.
    :return:
        - If return_episode_rewards is False:
              (mean_reward, std_reward, custom_info_dict)
          where custom_info_dict is a dict with keys:
              "mean_targets_reached", "avg_speed", "collision_rate"
        - If return_episode_rewards is True:
              (episode_rewards, episode_lengths, custom_info_dict)
          where each value in custom_info_dict is a list, one per episode.
    """
    # Ensure env is a VecEnv
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    # Check for Monitor wrapper
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a Monitor wrapper. "
            "This may result in modified episode lengths and rewards if other wrappers modify them. "
            "Consider wrapping the environment with Monitor.", UserWarning
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    # Custom info lists for each episode
    episode_mean_targets = []
    episode_mean_velocities = []
    episode_collision_ratios = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Distribute the number of episodes to evaluate among the sub-environments
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    agent = "drone0"

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i][agent]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    # If the Monitor wrapper is present, extract from the "episode" field.
                    if is_monitor_wrapped and "episode" in info.keys():
                        ep_info = info["episode"]
                        episode_rewards.append(ep_info["r"])
                        episode_lengths.append(ep_info["l"])
                        # Extract custom info if present
                        if "progress" in ep_info:
                            episode_mean_targets.append(ep_info["progress"])
                        if "velocity" in ep_info:
                            episode_mean_velocities.append(ep_info["velocity"])
                        if "collision" in ep_info:
                            episode_collision_ratios.append(ep_info["collision"])
                        episode_counts[i] += 1
                    else:
                        # For non-Monitor wrapped envs, use the current counters and info directly.
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        if "progress" in info:
                            episode_mean_targets.append(info["progress"])
                        if "velocity" in info:
                            episode_mean_velocities.append(info["velocity"])
                        if "collision" in info:
                            episode_collision_ratios.append(info["collision"])
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    if not return_episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        custom_info = {
            "mean_targets_reached": np.mean(episode_mean_targets) if episode_mean_targets else float("nan"),
            "mean_velocity": np.mean(episode_mean_velocities) if episode_mean_velocities else float("nan"),
            "mean_collision_ratio": np.mean(episode_collision_ratios) if episode_collision_ratios else float("nan"),
        }
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, (
                f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
            )
        return mean_reward, std_reward, custom_info
    else:
        custom_info = {
            "mean_targets_reached": episode_mean_targets,
            "mean_velocity": episode_mean_velocities,
            "mean_collision_ratio": episode_collision_ratios,
        }
        return episode_rewards, episode_lengths, custom_info



class CustomEvalCallback(EvalCallback):
    """
    Custom evaluation callback that logs additional metrics by extracting
    them from the environment's info dictionary. In particular, it logs:

      - target_success_rate (renamed from mean_targets_reached)
      - mean_velocity (renamed from avg_speed)
      - collision_frequency (renamed from collision_rate)

    The evaluation environment must include these keys in its `info` dict during evaluation.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(CustomEvalCallback, self).__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Synchronize normalization if using VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback"
                    ) from e

            # Reset success-rate buffer (for HER, for example)
            self._is_success_buffer = []

            # Evaluate the policy with extra infos returned:
            # Note: return_episode_infos must be True to get the extra info dict.
            result = evaluate_policy_updated(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            # Depending on the version of evaluate_policy, it might return 2 or 3 values.
            if len(result) == 3:
                episode_rewards, episode_lengths, episode_infos = result


            # Logging to disk (if log_path is provided)
            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if available
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            # Compute mean reward and timing metrics
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Log the standard metrics
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # If there is a success rate available (e.g., from HER), log it.
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Aggregate and log the additional custom metrics:
            target_success_rates: List[float] = []
            mean_velocities: List[float] = []
            collision_frequencies: List[float] = []

            # print ("episode_infos", episode_infos)

            if isinstance(episode_infos, dict):
                # Directly extract values from the dictionary.
                target_success_rates.extend(episode_infos.get("mean_targets_reached", []))
                mean_velocities.extend(episode_infos.get("mean_velocity", []))
                collision_frequencies.extend(episode_infos.get("mean_collision_ratio", []))
            else:
                for info in episode_infos:
                    # Sometimes info may be a list or dict
                    if isinstance(info, dict):
                        info_dict = info
                    elif isinstance(info, list) and len(info) > 0:
                        info_dict = info[0]
                    else:
                        info_dict = {}

                    # Extract target success rate (using the key "mean_targets_reached")
                    if "mean_targets_reached" in info_dict:
                        target_success_rates.append(info_dict["mean_targets_reached"])

                    # Extract mean velocity (using the key "mean_velocity")
                    if "mean_velocity" in info_dict:
                        mean_velocities.append(info_dict["mean_velocity"])

                    # Extract collision frequency (using the key "mean_collision_ratio")
                    if "mean_collision_ratio" in info_dict:
                        collision_frequencies.append(info_dict["mean_collision_ratio"])

            # Print the results to verify.
            # print("Target Success Rates:", target_success_rates)
            # print("Mean Velocities:", mean_velocities)
            # print("Collision Frequencies:", collision_frequencies)

            if len(target_success_rates):
                avg_target_success = np.mean(target_success_rates)
                self.logger.record("eval/target_success_rate", avg_target_success)
                # if self.verbose >= 1:
                #     print(f"Target success rate: {avg_target_success:.2f}")

            if len(mean_velocities):
                avg_mean_velocity = np.mean(mean_velocities)
                self.logger.record("eval/mean_velocity", avg_mean_velocity)
                # if self.verbose >= 1:
                #     print(f"Mean velocity: {avg_mean_velocity:.2e}")

            if len(collision_frequencies):
                avg_collision_frequency = np.mean(collision_frequencies)
                self.logger.record("eval/collision_frequency", avg_collision_frequency)
                # if self.verbose >= 1:
                #     print(f"Collision frequency: {avg_collision_frequency:.2e}")

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Save best model if current mean reward is the best so far.
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(model_path)
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables in the child callback.
        """
        if self.callback:
            self.callback.update_locals(locals_)