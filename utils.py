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
from collections import defaultdict
import pickle
import glob
from PIL import Image
import imageio
import subprocess


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
        self.gates = env.gates
        self.dt = env.dt
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


def evaluate_model_with_visual(model, env, n_episodes=5, max_steps=500, verbose=False, save_csv=False, 
                              main_folder="", visualize=False, visualization_folder=None, create_video=False):
    """
    Evaluate a given model in the provided environment with added visualization capability.
    Returns a Pandas DataFrame with metrics aggregated over episodes.
    """
    import matplotlib
    # Use Agg backend to avoid window management issues
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    
    episode_stats = []  # To store metrics per episode
    total_reward_list = []
    avg_speed_list = []
    targets_reached_list = []
    collisions_list = []
    drone_positions = []       

    # Get base environment and agents
    base_env = get_base_env_with_agents(env)
    if not hasattr(base_env, "agents"):
        raise AttributeError("The base environment does not have an 'agents' attribute.")
    agents = base_env.agents
    
    # Create visualization directory if needed
    if visualize and visualization_folder:
        os.makedirs(visualization_folder, exist_ok=True)
    
    # Set up visualization colors
    drone_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    gate_colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(base_env.gates)))
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        speeds = {}          # To store each agent's speeds over steps
        targets_reached = {}  # To store the final progress reported for each agent
        
        # For visualization tracking
        positions_history = []
        velocities_history = []
        gate_indices_history = []
        collision_history = []
        
        speeds = defaultdict(list)  # Automatically initializes lists
        targets_reached = defaultdict(float)  # Automatically initializes to 0.0
        drone_positions.append(defaultdict(list))  # Cleaner initialization

        # Run one episode.
        while not done and steps < max_steps:  # Add a step limit to prevent infinite loops
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Gather data for visualization
            step_positions = []
            step_velocities = []
            step_gate_indices = []
            step_collisions = []
            
            for agent_idx, agent in enumerate(agents):
                state = base_env.get_state(agent)
                speeds[agent].append(np.linalg.norm(state[3:6]))
                targets_reached[agent] = base_env.vehicles[agent]["progress"]
                drone_positions[episode][agent].append(state[0:3])
                
                # Extract position and velocity for visualization
                pos = state[0:3]
                vel = state[3:6]
                current_gate = base_env.vehicles[agent]["current_target_index"]
                
                step_positions.append(pos)
                step_velocities.append(vel)
                step_gate_indices.append(current_gate)
                
                # Check for collisions
                if "collision" in info[agent] and info[agent]["collision"]:
                    step_collisions.append((agent_idx, agent_idx))  # Format to match other visualization
            
            # Append step data to history
            positions_history.append(np.array(step_positions))
            velocities_history.append(np.array(step_velocities))
            gate_indices_history.append(step_gate_indices)
            if step_collisions:
                collision_history.append((steps, step_collisions))
            
            # Visualize each step if requested
            if visualize and (steps % 2 == 0 or done) and visualization_folder:  # Save every other step
                try:
                    # Close any existing figures to prevent memory leaks
                    plt.close('all')
                    
                    fig = visualize_step(
                        base_env, 
                        positions_history,
                        velocities_history,
                        gate_indices_history,
                        step_collisions,
                        drone_colors,
                        gate_colors,
                        steps,
                        total_reward,
                        safety_radius=base_env.env.drone_collision_margin
                    )
                    
                    plt.savefig(f"{visualization_folder}/step_{steps:03d}.png", dpi=80)
                    plt.close(fig)
                except Exception as e:
                    print(f"Visualization error: {e}")

        total_reward_list.append(total_reward)
        if verbose:
            print(f"Episode {episode+1}/{n_episodes}: Total Reward={total_reward:.2f}, Steps={steps}")

        avg_speed = {}
        for agent in agents:
            # Compute the average speed per agent.
            avg_speed[agent] = np.mean(speeds[agent])
            if verbose:
                print(f"Agent: {agent} Targets Reached={targets_reached[agent]}, Avg Speed={avg_speed[agent]:.2f}")
            avg_speed_list.append(avg_speed[agent])
            targets_reached_list.append(targets_reached[agent])
            collisions_list.append(1 if "collision" in info[agent] and info[agent]["collision"] else 0)

        episode_stats.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "targets_reached": targets_reached,
            "avg_speed": avg_speed,
            "collisions": sum(1 for a in agents if "collision" in info[a] and info[a]["collision"])
        })
        
        # Create video from frames if requested
        if visualize and create_video and visualization_folder:
            try:
                video_path = f"{visualization_folder}/episode_{episode}_video.mp4"
                create_video_from_frames(visualization_folder, video_path)
                
                # Clear frames to prepare for next episode
                if episode < n_episodes - 1:
                    import glob
                    for frame_file in glob.glob(f"{visualization_folder}/step_*.png"):
                        os.remove(frame_file)
            except Exception as e:
                print(f"Video creation error: {e}")

    # Calculate mean metrics
    mean_reward = np.mean(total_reward_list)
    mean_speed = np.mean(avg_speed_list)
    std_speed = np.std(avg_speed_list)
    mean_targets_reached = np.mean(targets_reached_list)
    std_targets_reached = np.std(targets_reached_list)
    mean_collisions = np.mean(collisions_list)
    std_collisions = np.std(collisions_list)

    print(f"\nEvaluation results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_speed:.2f}/±{std_speed:.2f}, "
          f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")

    with open(os.path.join(main_folder, "drone_positions.pkl"), "wb") as f:
        pickle.dump(drone_positions, f)

    if save_csv:
        stats_df = pd.DataFrame(episode_stats)
        results_csv = os.path.join(main_folder, "evaluation_results.csv")
        stats_df.to_csv(results_csv, index=False)

         

    return targets_reached_list, avg_speed_list, total_reward_list, collisions_list

def get_base_env_with_agents(env):
    """
    Recursively unwrap the environment until you find an object with an 'agents' attribute.
    """
    current_env = env
    while not hasattr(current_env, "agents") and hasattr(current_env, "env"):
        current_env = current_env.env
    return current_env

def visualize_step(env, positions_history, velocities_history, gate_indices_history, 
                 collisions, drone_colors, gate_colors, current_step, total_reward, safety_radius:float = 0.5):
    """
    Visualize the current simulation state in a similar style to the IBR approach.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    
    # Create figure with explicit renderer
    fig = plt.figure(figsize=(12, 10), dpi=80)
    gs = gridspec.GridSpec(3, 3)
    
    # 3D view of the environment
    ax_3d = fig.add_subplot(gs[:2, :], projection='3d')
    
    # Status panel
    ax_status = fig.add_subplot(gs[2, 0])
    ax_status.axis('off')
    
    # Top-down view
    ax_top = fig.add_subplot(gs[2, 1])
    
    # Progress plot
    ax_progress = fig.add_subplot(gs[2, 2])
    
    # Set up axes
    max_bounds = 10.0  # Set an appropriate boundary
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('Multi-Drone Gate Navigation (RL)', fontsize=14)
    ax_3d.set_xlim(-max_bounds, max_bounds)
    ax_3d.set_ylim(-max_bounds, max_bounds)
    ax_3d.set_zlim(0, max_bounds)
    
    ax_top.set_title('Top-Down View', fontsize=10)
    ax_top.set_xlabel('X (m)')
    ax_top.set_ylabel('Y (m)')
    ax_top.set_xlim(-max_bounds, max_bounds)
    ax_top.set_ylim(-max_bounds, max_bounds)
    ax_top.set_aspect('equal')
    ax_top.grid(True, alpha=0.3)
    
    ax_progress.set_title('Gate Progress', fontsize=10)
    ax_progress.set_xlabel('Simulation Step')
    ax_progress.set_ylabel('Gate Index')
    ax_progress.set_xlim(0, 100)  # Adjust as needed
    ax_progress.set_ylim(-0.5, len(env.gates) - 0.5)
    ax_progress.grid(True, alpha=0.3)
    
    # Draw world environment
    # Draw circular boundary
    theta = np.linspace(0, 2*np.pi, 50)
    world_radius = 10.0  # Set an appropriate radius
    x = world_radius * np.cos(theta)
    y = world_radius * np.sin(theta)
    z = np.zeros_like(theta)
    
    ax_3d.plot(x, y, z, color="k", alpha=0.3, linewidth=1)
    ax_top.plot(x, y, color="k", alpha=0.3, linewidth=1)
    
    # Draw ground plane
    x_grid = np.linspace(-world_radius, world_radius, 5)
    y_grid = np.linspace(-world_radius, world_radius, 5)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    ax_3d.plot_surface(X, Y, Z, color='gray', alpha=0.1, edgecolor='none')
    
    # Draw gates
    for i, gate in enumerate(env.gates):
        try:
            # Check if gate is a dictionary (as in the previous code)
            if isinstance(gate, dict):
                center = gate["position"] if "position" in gate else gate["center"]
                yaw = gate["yaw"]
            else:
                # Assume gate is an object with attributes
                center = gate.position if hasattr(gate, "position") else gate.center
                yaw = gate.yaw
            
            # Create gate circle
            gate_radius = 1.0
            theta = np.linspace(0, 2*np.pi, 30)
            circle_x = gate_radius * np.cos(theta)
            circle_y = gate_radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            
            # Rotate and translate
            rot_x = np.cos(yaw) * circle_x - np.sin(yaw) * circle_y
            rot_y = np.sin(yaw) * circle_x + np.cos(yaw) * circle_y
            
            gate_x = center[0] + rot_x
            gate_y = center[1] + rot_y
            gate_z = center[2] + circle_z
            
            # Determine gate color
            gate_color = gate_colors[i]
            alpha = 0.5
            linewidth = 1
            
            # Check if this gate is a current target for any drone
            active = False
            if len(gate_indices_history) > 0:
                current_gates = gate_indices_history[-1]
                if i in current_gates:
                    active = True
                    alpha = 0.8
                    linewidth = 2
            
            # Draw gate in 3D view
            ax_3d.plot(gate_x, gate_y, gate_z, '-', color=gate_color, 
                      linewidth=linewidth, alpha=alpha)
            
            # Draw gate normal vector
            normal_length = 1.5
            normal_x = [center[0], center[0] + normal_length * np.cos(yaw)]
            normal_y = [center[1], center[1] + normal_length * np.sin(yaw)]
            normal_z = [center[2], center[2]]
            ax_3d.plot(normal_x, normal_y, normal_z, ':', color=gate_color, 
                      linewidth=linewidth, alpha=0.6)
            
            # Draw gate in top-down view
            ax_top.plot(gate_x, gate_y, '-', color=gate_color, 
                       linewidth=linewidth, alpha=alpha)
            
            # Add gate label
            ax_3d.text(center[0], center[1], center[2]+1.2, f"G{i}", 
                     color='red' if active else 'darkred', fontsize=10, 
                     ha='center', weight='bold' if active else 'normal')
            
            ax_top.text(center[0], center[1], f"G{i}", 
                      color='red' if active else 'darkred', fontsize=9, 
                      ha='center', weight='bold' if active else 'normal')
        except Exception as e:
            print(f"Error drawing gate {i}: {e}")
    
    # Draw drones
    if len(positions_history) > 0:
        positions = positions_history[-1]
        velocities = velocities_history[-1]
        
        # Set up collided drones
        collided_drones = set()
        if collisions:
            for collision in collisions:
                collided_drones.add(collision[0])
                collided_drones.add(collision[1])
        
        # Draw each drone
        for i, agent in enumerate(env.agents):
            try:
                pos = positions[i]
                vel = velocities[i]
                
                # Get current gate index for this drone
                if len(gate_indices_history) > 0:
                    current_gate_idx = gate_indices_history[-1][i]
                else:
                    current_gate_idx = env.vehicles[agent]["current_target_index"]
                
                # Make sure gate index is within bounds
                current_gate_idx = min(current_gate_idx, len(env.gates) - 1)
                
                # Get gate position
                gate = env.gates[current_gate_idx]
                if isinstance(gate, dict):
                    gate_pos = gate["position"] if "position" in gate else gate["center"]
                else:
                    gate_pos = gate.position if hasattr(gate, "position") else gate.center
                
                # Determine drone color and style
                drone_color = drone_colors[i % len(drone_colors)]
                marker_size = 100
                edge_width = 1
                
                # Highlight drones involved in collisions
                if i in collided_drones:
                    edge_width = 3
                    marker_size = 150
                
                # 3D view - drone position
                ax_3d.scatter(pos[0], pos[1], pos[2], color=drone_color, 
                             s=marker_size, edgecolors='red' if i in collided_drones else 'black',
                             linewidths=edge_width, label=f'D{i}→G{current_gate_idx}')
                
                # Top-down view - drone position
                ax_top.scatter(pos[0], pos[1], color=drone_color, 
                              s=marker_size*0.7, edgecolors='red' if i in collided_drones else 'black',
                              linewidths=edge_width)
                
                # Draw safety radius for collided drones
                if i in collided_drones:
                    # Create a circle in top-down view
                    theta = np.linspace(0, 2*np.pi, 20)
                    circle_x = pos[0] + safety_radius * np.cos(theta)
                    circle_y = pos[1] + safety_radius * np.sin(theta)
                    ax_top.plot(circle_x, circle_y, '--', color='red', alpha=0.4)
                
                # Draw velocity vectors
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 0.1:
                    vel_scale = 0.7  # Scale factor for velocity vector
                    vel_x = [pos[0], pos[0] + vel[0]/vel_norm * vel_scale]
                    vel_y = [pos[1], pos[1] + vel[1]/vel_norm * vel_scale]
                    vel_z = [pos[2], pos[2] + vel[2]/vel_norm * vel_scale]
                    
                    ax_3d.plot(vel_x, vel_y, vel_z, '->', color=drone_color, 
                              linewidth=2, alpha=0.8)
                    ax_top.plot([pos[0], pos[0] + vel[0]/vel_norm * vel_scale],
                               [pos[1], pos[1] + vel[1]/vel_norm * vel_scale],
                               '->', color=drone_color, linewidth=2, alpha=0.8)
                
                # Draw line to current target gate
                ax_3d.plot([pos[0], gate_pos[0]], [pos[1], gate_pos[1]], 
                          [pos[2], gate_pos[2]], color=drone_color, 
                          linestyle=':', alpha=0.3)
                ax_top.plot([pos[0], gate_pos[0]], [pos[1], gate_pos[1]], 
                           color=drone_color, linestyle=':', alpha=0.3)
            except Exception as e:
                print(f"Error drawing drone {i}: {e}")
    
    # Draw trajectories (last few positions)
    trajectory_length = min(10, len(positions_history))
    if trajectory_length > 1:
        for i, agent in enumerate(env.agents):
            try:
                color = drone_colors[i % len(drone_colors)]
                traj = np.array([positions_history[-j][i] for j in range(1, trajectory_length+1)][::-1])
                
                if len(traj) > 1:
                    ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                              color=color, linewidth=1.5, alpha=0.6)
                    ax_top.plot(traj[:, 0], traj[:, 1], 
                               color=color, linewidth=1.5, alpha=0.6)
            except Exception as e:
                print(f"Error drawing trajectory for drone {i}: {e}")
    
    # Update progress plot
    if len(gate_indices_history) > 0:
        try:
            # Convert gate indices to numpy array
            progress_data = np.array(gate_indices_history)
            steps = np.arange(len(gate_indices_history))
            
            for i, agent in enumerate(env.agents):
                color = drone_colors[i % len(drone_colors)]
                ax_progress.step(steps, progress_data[:, i], where='post', 
                                color=color, linewidth=1.5, alpha=0.7, label=f'Drone {i}')
            
            # Add vertical line for current step
            ax_progress.axvline(x=len(steps)-1, color='gray', linestyle='--', alpha=0.5)
        except Exception as e:
            print(f"Error plotting progress: {e}")
    
    # Add legend to progress plot
    ax_progress.legend(loc='upper left', fontsize=8)
    
    # Adjust tick marks to match gate indices
    ax_progress.set_yticks(range(len(env.gates)))
    ax_progress.set_yticklabels([f'G{i}' for i in range(len(env.gates))])
    
    # Update status panel
    ax_status.clear()
    ax_status.axis('off')
    
    # Title
    ax_status.text(0.05, 0.95, "STATUS", fontsize=12, weight='bold')
    
    # Simulation info
    sim_time = current_step * env.dt if hasattr(env, 'dt') else current_step * 0.1  # Default to 0.1s if dt not available
    ax_status.text(0.05, 0.90, f"Time: {sim_time:.1f}s", fontsize=10)
    ax_status.text(0.05, 0.85, f"Step: {current_step}", fontsize=10)
    ax_status.text(0.05, 0.80, f"Reward: {total_reward:.2f}", fontsize=10)
    
    collision_count = len(collided_drones)
    ax_status.text(0.05, 0.75, f"Collisions: {collision_count}", fontsize=10, 
                 color='red' if collision_count > 0 else 'black')
    
    # Drone info
    if len(positions_history) > 0:
        positions = positions_history[-1]
        velocities = velocities_history[-1]
        
        y_offset = 0.70
        for i, agent in enumerate(env.agents):
            try:
                color = drone_colors[i % len(drone_colors)]
                pos = positions[i]
                vel = velocities[i]
                speed = np.linalg.norm(vel)
                
                if len(gate_indices_history) > 0:
                    gate_idx = gate_indices_history[-1][i]
                else:
                    gate_idx = env.vehicles[agent]["current_target_index"]
                
                # Make sure gate index is within bounds
                gate_idx = min(gate_idx, len(env.gates) - 1)
                
                # Get gate position
                gate = env.gates[gate_idx]
                if isinstance(gate, dict):
                    gate_pos = gate["position"] if "position" in gate else gate["center"]
                else:
                    gate_pos = gate.position if hasattr(gate, "position") else gate.center
                
                dist_to_gate = np.linalg.norm(pos - gate_pos)
                
                # Add drone info
                ax_status.text(0.05, y_offset, f"Drone {i}", fontsize=10, weight='bold', color=color)
                ax_status.text(0.05, y_offset-0.04, f"Gate: {gate_idx}", fontsize=9)
                ax_status.text(0.05, y_offset-0.08, f"Dist: {dist_to_gate:.2f}m", fontsize=9)
                ax_status.text(0.05, y_offset-0.12, f"Speed: {speed:.2f}m/s", fontsize=9)
                
                y_offset -= 0.15
            except Exception as e:
                print(f"Error updating status for drone {i}: {e}")
    
    # Adjust plot layout
    plt.tight_layout()
    
    return fig


def create_video_from_frames(frame_directory, output_file="simulation_video.mp4", fps=10):
    """Create a video from a directory of frame images"""
    try:
        import cv2
        import os
        import glob
        
        # Find all frame files
        frame_files = sorted(glob.glob(os.path.join(frame_directory, "step_*.png")))
        
        if not frame_files:
            print(f"No frame files found in {frame_directory}")
            return None
        
        print(f"Creating video from {len(frame_files)} frames...")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Add each frame to the video
        for frame_file in frame_files:
            video.write(cv2.imread(frame_file))
        
        # Release the video writer
        video.release()
        
        print(f"Video created successfully: {output_file}")
        return output_file
    
    except ImportError:
        print("Error: OpenCV (cv2) is required to create videos.")
        print("Install it with: pip install opencv-python")
        return None
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return None
    



def evaluate_model_with_visual_v2(model, env, n_episodes=5, max_steps=500, verbose=False, save_csv=False, 
                              main_folder="", visualize=False, visualization_folder=None, create_video=False):
    """
    Evaluate a given model in the provided environment with enhanced visualization for journal publication.
    Returns a Pandas DataFrame with metrics aggregated over episodes.
    """
    import matplotlib
    # Use Agg backend to avoid window management issues
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    from collections import defaultdict
    import pickle
    from matplotlib import cm
    import matplotlib.colors as mcolors
    
    # Set global plotting style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['figure.dpi'] = 120
    
    episode_stats = []  # To store metrics per episode
    total_reward_list = []
    avg_speed_list = []
    targets_reached_list = []
    collisions_list = []
    drone_positions = []

    # Get base environment and agents
    base_env = get_base_env_with_agents(env)
    if not hasattr(base_env, "agents"):
        raise AttributeError("The base environment does not have an 'agents' attribute.")
    agents = base_env.agents
    
    # Create visualization directory if needed
    if visualize and visualization_folder:
        os.makedirs(visualization_folder, exist_ok=True)
    
    # Set up visualization colors - using a professional color palette
    # Create a custom color palette for drones that's distinct and professional
    custom_drone_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    drone_colors = [mcolors.to_rgba(c) for c in custom_drone_colors]
    
    # Use a sequential colormap for gates with more subdued colors
    gate_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(base_env.gates)))
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        # Initialize tracking containers
        speeds = defaultdict(list)  # Automatically initializes lists
        targets_reached = defaultdict(float)  # Automatically initializes to 0.0
        drone_positions.append(defaultdict(list))
        
        # For visualization tracking
        positions_history = []
        velocities_history = []
        gate_indices_history = []
        collision_history = []
        altitude_history = defaultdict(list)  # Track altitude for each agent
        speed_history = defaultdict(list)     # Track speed for each agent

        # Run one episode.
        while not done and steps < max_steps:  # Add a step limit to prevent infinite loops
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Gather data for visualization
            step_positions = []
            step_velocities = []
            step_gate_indices = []
            step_collisions = []
            
            for agent_idx, agent in enumerate(agents):
                state = base_env.get_state(agent)
                current_speed = np.linalg.norm(state[3:6])
                speeds[agent].append(current_speed)
                speed_history[agent].append(current_speed)
                targets_reached[agent] = base_env.vehicles[agent]["progress"]
                drone_positions[episode][agent].append(state[0:3])
                altitude_history[agent].append(state[2])  # Z coordinate for altitude
                
                # Extract position and velocity for visualization
                pos = state[0:3]
                vel = state[3:6]
                current_gate = base_env.vehicles[agent]["current_target_index"]
                
                step_positions.append(pos)
                step_velocities.append(vel)
                step_gate_indices.append(current_gate)
                
                # Check for collisions
                if "collision" in info[agent] and info[agent]["collision"]:
                    step_collisions.append((agent_idx, agent_idx))
            
            # Append step data to history
            positions_history.append(np.array(step_positions))
            velocities_history.append(np.array(step_velocities))
            gate_indices_history.append(step_gate_indices)
            if step_collisions:
                collision_history.append((steps, step_collisions))
            
            # Visualize each step if requested
            if visualize and (steps % 2 == 0 or done) and visualization_folder:
                try:
                    # Close any existing figures to prevent memory leaks
                    plt.close('all')
                    
                    fig = visualize_step_enhanced(
                        base_env, 
                        positions_history,
                        velocities_history,
                        gate_indices_history,
                        collision_history,
                        drone_colors,
                        gate_colors,
                        steps,
                        total_reward,
                        altitude_history,
                        speed_history,
                        safety_radius=base_env.env.drone_collision_margin
                    )
                    
                    plt.savefig(f"{visualization_folder}/step_{steps:03d}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Visualization error: {e}")

        total_reward_list.append(total_reward)
        if verbose:
            print(f"Episode {episode+1}/{n_episodes}: Total Reward={total_reward:.2f}, Steps={steps}")

        avg_speed = {}
        for agent in agents:
            # Compute the average speed per agent.
            avg_speed[agent] = np.mean(speeds[agent])
            if verbose:
                print(f"Agent: {agent} Targets Reached={targets_reached[agent]}, Avg Speed={avg_speed[agent]:.2f}")
            avg_speed_list.append(avg_speed[agent])
            targets_reached_list.append(targets_reached[agent])
            collisions_list.append(1 if "collision" in info[agent] and info[agent]["collision"] else 0)

        episode_stats.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "targets_reached": targets_reached,
            "avg_speed": avg_speed,
            "collisions": sum(1 for a in agents if "collision" in info[a] and info[a]["collision"])
        })
        
        # Create video from frames if requested
        if visualize and create_video and visualization_folder:
            try:
                video_path = f"{visualization_folder}/episode_{episode}_video.mp4"
                create_video_from_frames_v2(visualization_folder, video_path, fps=15)  # Higher FPS for smoother video
                
                # Clear frames to prepare for next episode
                if episode < n_episodes - 1:
                    import glob
                    for frame_file in glob.glob(f"{visualization_folder}/step_*.png"):
                        os.remove(frame_file)
            except Exception as e:
                print(f"Video creation error: {e}")

    # Calculate mean metrics
    mean_reward = np.mean(total_reward_list)
    mean_speed = np.mean(avg_speed_list)
    std_speed = np.std(avg_speed_list)
    mean_targets_reached = np.mean(targets_reached_list)
    std_targets_reached = np.std(targets_reached_list)
    mean_collisions = np.mean(collisions_list)
    std_collisions = np.std(collisions_list)

    print(f"\nEvaluation results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_speed:.2f}/±{std_speed:.2f}, "
          f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")

    with open(os.path.join(main_folder, "drone_positions.pkl"), "wb") as f:
        pickle.dump(drone_positions, f)

    if save_csv:
        stats_df = pd.DataFrame(episode_stats)
        results_csv = os.path.join(main_folder, "evaluation_results.csv")
        stats_df.to_csv(results_csv, index=False)

    return targets_reached_list, avg_speed_list, total_reward_list, collisions_list


def visualize_step_enhanced(env, positions_history, velocities_history, gate_indices_history, 
                           collision_history, drone_colors, gate_colors, current_step, total_reward, 
                           altitude_history, speed_history, safety_radius=0.5):
    """
    Enhanced visualization of drone simulation for journal publication quality.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.colors as colors
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as path_effects
    
    class Arrow3D(FancyArrowPatch):
        """Custom 3D arrow for better visualization"""
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)
    
    # Create figure with professional layout
    fig = plt.figure(figsize=(15, 12), dpi=150, facecolor='white')
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1.5, 1.5, 1, 1])
    
    # 3D view of the environment
    ax_3d = fig.add_subplot(gs[:2, :2], projection='3d')
    
    # Create telemetry panels
    ax_top = fig.add_subplot(gs[0, 2])      # Top-down view
    ax_side = fig.add_subplot(gs[0, 3])     # Side view
    ax_progress = fig.add_subplot(gs[1, 2]) # Gate progress graph
    ax_status = fig.add_subplot(gs[1, 3])   # Status display
    
    # Bottom row for additional metrics
    ax_altitude = fig.add_subplot(gs[2, 0]) # Altitude over time
    ax_speed = fig.add_subplot(gs[2, 1])    # Speed over time
    ax_info = fig.add_subplot(gs[2, 2:])    # Simulation info
    
    # Set background colors for professional appearance
    for ax in [ax_3d, ax_top, ax_side, ax_progress, ax_status, ax_altitude, ax_speed, ax_info]:
        ax.set_facecolor('#f8f8f8')
    
    # Set up axes with consistent styling
    max_bounds = 10.0  # Set appropriate boundary
    
    # Common styling function
    def style_axes(ax, title, xlabel=None, ylabel=None, zlabel=None):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
        if zlabel:
            ax.set_zlabel(zlabel, fontsize=10, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        # Add subtle grid
        ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    
    # 3D View styling
    style_axes(ax_3d, 'Multi-Drone Gate Navigation (3D View)', 'X (m)', 'Y (m)', 'Z (m)')
    ax_3d.set_xlim(-max_bounds, max_bounds)
    ax_3d.set_ylim(-max_bounds, max_bounds)
    ax_3d.set_zlim(0, max_bounds)
    # Improve 3D perspective
    ax_3d.view_init(elev=30, azim=45)
    
    # Top view styling
    style_axes(ax_top, 'Top-Down View', 'X (m)', 'Y (m)')
    ax_top.set_xlim(-max_bounds, max_bounds)
    ax_top.set_ylim(-max_bounds, max_bounds)
    ax_top.set_aspect('equal')
    
    # Side view styling
    style_axes(ax_side, 'Side View', 'X (m)', 'Z (m)')
    ax_side.set_xlim(-max_bounds, max_bounds)
    ax_side.set_ylim(0, max_bounds)
    
    # Progress graph styling
    style_axes(ax_progress, 'Gate Progress Timeline', 'Simulation Step', 'Gate Index')
    ax_progress.set_xlim(0, max(100, current_step + 10))
    ax_progress.set_ylim(-0.5, len(env.gates) - 0.5)
    
    # Altitude graph styling
    style_axes(ax_altitude, 'Altitude Over Time', 'Simulation Step', 'Altitude (m)')
    ax_altitude.set_xlim(0, max(100, current_step + 10))
    ax_altitude.set_ylim(0, max_bounds)
    
    # Speed graph styling
    style_axes(ax_speed, 'Speed Profile', 'Simulation Step', 'Speed (m/s)')
    ax_speed.set_xlim(0, max(100, current_step + 10))
    max_speed = 5.0  # Adjust based on your environment
    ax_speed.set_ylim(0, max_speed)
    
    # Status and info displays have no axes
    ax_status.axis('off')
    ax_info.axis('off')
    
    # Draw world environment
    # Create a more visually appealing boundary
    theta = np.linspace(0, 2*np.pi, 100)
    world_radius = 10.0
    x = world_radius * np.cos(theta)
    y = world_radius * np.sin(theta)
    z = np.zeros_like(theta)
    
    # Draw circular boundary with better styling
    ax_3d.plot(x, y, z, color="#555555", alpha=0.4, linewidth=1, linestyle='-')
    ax_top.plot(x, y, color="#555555", alpha=0.4, linewidth=1, linestyle='-')
    
    # Draw ground plane with gradient for better depth perception
    x_grid = np.linspace(-world_radius, world_radius, 20)
    y_grid = np.linspace(-world_radius, world_radius, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    # Create a custom colormap for the ground
    ground_colors = [(0.95, 0.95, 0.95, 0.1), (0.85, 0.85, 0.85, 0.3)]
    ground_cmap = LinearSegmentedColormap.from_list("ground_cmap", ground_colors)
    
    # Draw ground with distance-based fading
    ground_color = np.sqrt(X**2 + Y**2) / world_radius
    ax_3d.plot_surface(X, Y, Z, facecolors=ground_cmap(ground_color), 
                      rstride=1, cstride=1, linewidth=0.5, alpha=0.2)
    
    # Add coordinate axes for reference
    # Origin
    origin = [0, 0, 0]
    # X axis
    x_axis = Arrow3D([0, 2], [0, 0], [0, 0], mutation_scale=15, 
                    lw=2, arrowstyle='-|>', color='red')
    ax_3d.add_artist(x_axis)
    ax_3d.text(2.2, 0, 0, "X", color='red', fontsize=10)
    
    # Y axis
    y_axis = Arrow3D([0, 0], [0, 2], [0, 0], mutation_scale=15, 
                    lw=2, arrowstyle='-|>', color='green')
    ax_3d.add_artist(y_axis)
    ax_3d.text(0, 2.2, 0, "Y", color='green', fontsize=10)
    
    # Z axis
    z_axis = Arrow3D([0, 0], [0, 0], [0, 2], mutation_scale=15, 
                    lw=2, arrowstyle='-|>', color='blue')
    ax_3d.add_artist(z_axis)
    ax_3d.text(0, 0, 2.2, "Z", color='blue', fontsize=10)
    
    # Draw gates
    for i, gate in enumerate(env.gates):
        try:
            # Check if gate is a dictionary (as in the previous code)
            if isinstance(gate, dict):
                center = gate["position"] if "position" in gate else gate["center"]
                yaw = gate["yaw"]
            else:
                # Assume gate is an object with attributes
                center = gate.position if hasattr(gate, "position") else gate.center
                yaw = gate.yaw
            
            # Create gate circle
            gate_radius = 1.0
            theta = np.linspace(0, 2*np.pi, 50)  # More points for smoother circles
            circle_x = gate_radius * np.cos(theta)
            circle_y = gate_radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            
            # Rotate and translate
            rot_x = np.cos(yaw) * circle_x - np.sin(yaw) * circle_y
            rot_y = np.sin(yaw) * circle_x + np.cos(yaw) * circle_y
            
            gate_x = center[0] + rot_x
            gate_y = center[1] + rot_y
            gate_z = center[2] + circle_z
            
            # Enhanced styling for gates
            gate_color = gate_colors[i]
            alpha = 0.5
            linewidth = 1.5
            
            # Check if this gate is a current target for any drone
            active = False
            if len(gate_indices_history) > 0:
                current_gates = gate_indices_history[-1]
                if i in current_gates:
                    active = True
                    alpha = 0.85
                    linewidth = 2.5
            
            # Draw gate in 3D view with glow effect for active gates
            if active:
                # Draw a subtle glow around active gates
                glow = path_effects.withStroke(linewidth=4, foreground='yellow', alpha=0.3)
                line = ax_3d.plot(gate_x, gate_y, gate_z, '-', color=gate_color, 
                                 linewidth=linewidth, alpha=alpha)[0]
                line.set_path_effects([glow])
            else:
                ax_3d.plot(gate_x, gate_y, gate_z, '-', color=gate_color, 
                          linewidth=linewidth, alpha=alpha)
            
            # Draw gate normal vector with arrow
            normal_length = 1.5
            
            normal_arrow = Arrow3D([center[0], center[0] + normal_length * np.cos(yaw)],
                                 [center[1], center[1] + normal_length * np.sin(yaw)],
                                 [center[2], center[2]],
                                 mutation_scale=10, lw=1.5, arrowstyle='-|>', color=gate_color, alpha=0.7)
            ax_3d.add_artist(normal_arrow)
            
            # Draw gate in top-down view
            if active:
                # Draw a subtle glow around active gates
                glow = path_effects.withStroke(linewidth=4, foreground='yellow', alpha=0.3)
                line = ax_top.plot(gate_x, gate_y, '-', color=gate_color, 
                                  linewidth=linewidth, alpha=alpha)[0]
                line.set_path_effects([glow])
            else:
                ax_top.plot(gate_x, gate_y, '-', color=gate_color, 
                           linewidth=linewidth, alpha=alpha)
            
            # Draw gate in side view
            ax_side.plot([center[0]-gate_radius, center[0]+gate_radius], [center[2], center[2]], 
                        '-', color=gate_color, linewidth=linewidth, alpha=alpha)
            
            # Add gate label with better styling
            text_color = 'red' if active else '#880000'
            fontweight = 'bold' if active else 'normal'
            
            gate_text = ax_3d.text(center[0], center[1], center[2]+1.2, f"G{i}", 
                                  color=text_color, fontsize=10, ha='center', weight=fontweight)
            
            # Add shadow effect to text for better visibility
            gate_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            
            # Add gate labels to other views
            ax_top.text(center[0], center[1], f"G{i}", color=text_color, fontsize=9, 
                       ha='center', weight=fontweight,
                       path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
            
            ax_side.text(center[0], center[2]+0.3, f"G{i}", color=text_color, fontsize=9, 
                        ha='center', weight=fontweight,
                        path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        except Exception as e:
            print(f"Error drawing gate {i}: {e}")
    
    # Draw drones
    if len(positions_history) > 0:
        positions = positions_history[-1]
        velocities = velocities_history[-1]
        
        # Set up collided drones
        collided_drones = set()
        for collision in collision_history:
            if collision[0] <= current_step:  # Only show active collisions
                for drone_pair in collision[1]:
                    collided_drones.add(drone_pair[0])
                    collided_drones.add(drone_pair[1])
        
        # Draw trails before drones for better layering
        trajectory_length = min(15, len(positions_history))
        if trajectory_length > 1:
            for i, agent in enumerate(env.agents):
                try:
                    color = drone_colors[i % len(drone_colors)]
                    traj = np.array([positions_history[-j][i] for j in range(1, trajectory_length+1)][::-1])
                    
                    if len(traj) > 1:
                        # Create fading effect for trajectory
                        for j in range(len(traj)-1):
                            alpha = 0.3 + 0.5 * (j / len(traj))  # Fade from less visible to more visible
                            ax_3d.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2], 
                                      color=color, linewidth=1.5, alpha=alpha)
                            ax_top.plot(traj[j:j+2, 0], traj[j:j+2, 1], 
                                       color=color, linewidth=1.5, alpha=alpha)
                            ax_side.plot(traj[j:j+2, 0], traj[j:j+2, 2], 
                                        color=color, linewidth=1.5, alpha=alpha)
                except Exception as e:
                    print(f"Error drawing trajectory for drone {i}: {e}")
        
        # Draw each drone with enhanced styling
        for i, agent in enumerate(env.agents):
            try:
                pos = positions[i]
                vel = velocities[i]
                
                # Get current gate index for this drone
                if len(gate_indices_history) > 0:
                    current_gate_idx = gate_indices_history[-1][i]
                else:
                    current_gate_idx = env.vehicles[agent]["current_target_index"]
                
                # Make sure gate index is within bounds
                current_gate_idx = min(current_gate_idx, len(env.gates) - 1)
                
                # Get gate position
                gate = env.gates[current_gate_idx]
                if isinstance(gate, dict):
                    gate_pos = gate["position"] if "position" in gate else gate["center"]
                else:
                    gate_pos = gate.position if hasattr(gate, "position") else gate.center
                
                # Determine drone color and style
                drone_color = drone_colors[i % len(drone_colors)]
                marker_size = 120
                edge_width = 1.5
                
                # Highlight drones involved in collisions
                if i in collided_drones:
                    edge_width = 3
                    marker_size = 180
                    edge_color = 'red'
                    # Add warning glow
                    glow_effect = path_effects.withStroke(linewidth=5, foreground='red', alpha=0.3)
                else:
                    edge_color = 'black'
                    glow_effect = None
                
                # 3D view - drone position with enhanced markers
                drone_marker = ax_3d.scatter(pos[0], pos[1], pos[2], color=drone_color, 
                                           s=marker_size, edgecolors=edge_color,
                                           linewidths=edge_width, label=f'D{i}→G{current_gate_idx}',
                                           marker='o', alpha=0.9)
                
                # Add drone label
                drone_label = ax_3d.text(pos[0], pos[1], pos[2]+0.3, f"D{i}", 
                                       color='white', fontsize=8, ha='center', weight='bold',
                                       bbox=dict(facecolor=drone_color, alpha=0.7, pad=1, boxstyle='round'))
                
                # Top-down view - drone position
                ax_top.scatter(pos[0], pos[1], color=drone_color, 
                              s=marker_size*0.7, edgecolors=edge_color,
                              linewidths=edge_width)
                
                # Side view - drone position
                ax_side.scatter(pos[0], pos[2], color=drone_color, 
                               s=marker_size*0.7, edgecolors=edge_color,
                               linewidths=edge_width)
                
                # Add drone labels to 2D views
                ax_top.text(pos[0], pos[1]+0.3, f"D{i}", color='white', fontsize=7, ha='center',
                           bbox=dict(facecolor=drone_color, alpha=0.7, pad=1, boxstyle='round'))
                
                ax_side.text(pos[0], pos[2]+0.3, f"D{i}", color='white', fontsize=7, ha='center',
                            bbox=dict(facecolor=drone_color, alpha=0.7, pad=1, boxstyle='round'))
                
                # Draw safety radius for collided drones
                if i in collided_drones:
                    # Create a circle in top-down view
                    theta = np.linspace(0, 2*np.pi, 30)
                    circle_x = pos[0] + safety_radius * np.cos(theta)
                    circle_y = pos[1] + safety_radius * np.sin(theta)
                    ax_top.plot(circle_x, circle_y, '--', color='red', alpha=0.6, linewidth=1.5)
                    
                    # Create a visual indicator in 3D
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = pos[0] + safety_radius * np.cos(u) * np.sin(v)
                    y = pos[1] + safety_radius * np.sin(u) * np.sin(v)
                    z = pos[2] + safety_radius * np.cos(v)
                    ax_3d.plot_wireframe(x, y, z, color='red', alpha=0.2, linewidth=0.5)
                
                # Draw velocity vectors with fancier arrows
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 0.1:
                    vel_scale = 1.0  # Scale factor for velocity vector
                    scaled_vel = vel / vel_norm * vel_scale
                    
                    # 3D velocity arrow
                    vel_arrow = Arrow3D([pos[0], pos[0] + scaled_vel[0]],
                                      [pos[1], pos[1] + scaled_vel[1]],
                                      [pos[2], pos[2] + scaled_vel[2]],
                                      mutation_scale=12, lw=2, arrowstyle='-|>', color=drone_color, alpha=0.8)
                    ax_3d.add_artist(vel_arrow)
                    
                    # 2D velocity arrows
                    ax_top.arrow(pos[0], pos[1], scaled_vel[0], scaled_vel[1], 
                               head_width=0.2, head_length=0.3, fc=drone_color, ec=drone_color, alpha=0.8)
                    
                    ax_side.arrow(pos[0], pos[2], scaled_vel[0], scaled_vel[2], 
                                head_width=0.2, head_length=0.3, fc=drone_color, ec=drone_color, alpha=0.8)
                
                # Draw line to current target gate with better styling
                ax_3d.plot([pos[0], gate_pos[0]], [pos[1], gate_pos[1]], 
                          [pos[2], gate_pos[2]], color=drone_color, 
                          linestyle='--', alpha=0.4, linewidth=1)
                
                ax_top.plot([pos[0], gate_pos[0]], [pos[1], gate_pos[1]], 
                           color=drone_color, linestyle='--', alpha=0.4, linewidth=1)
                
                ax_side.plot([pos[0], gate_pos[0]], [pos[2], gate_pos[2]], 
                            color=drone_color, linestyle='--', alpha=0.4, linewidth=1)
                
                # Calculate distance to gate
                dist_to_gate = np.linalg.norm(np.array(pos) - np.array(gate_pos))
                
                # Add distance indicator
                mid_x = (pos[0] + gate_pos[0]) / 2
                mid_y = (pos[1] + gate_pos[1]) / 2
                mid_z = (pos[2] + gate_pos[2]) / 2
                
                # Only show distance for the closest gate
                if i == 0:  # Just for main drone to avoid clutter
                    ax_3d.text(mid_x, mid_y, mid_z, f"{dist_to_gate:.1f}m", 
                             color=drone_color, fontsize=8, ha='center',
                             path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
            except Exception as e:
                print(f"Error drawing drone {i}: {e}")
    
    # Update progress plot with enhanced styling
    if len(gate_indices_history) > 0:
        try:
            # Convert gate indices to numpy array
            progress_data = np.array(gate_indices_history)
            steps = np.arange(len(gate_indices_history))
            
            # Draw with enhanced styling
            for i, agent in enumerate(env.agents):
                color = drone_colors[i % len(drone_colors)]
                # Add solid marker at each gate change for better visibility
                change_points = [0]
                
                for j in range(1, len(progress_data)):
                    if progress_data[j, i] != progress_data[j-1, i]:
                        change_points.append(j)
                
                # Plot the step line
                ax_progress.step(steps, progress_data[:, i], where='post', 
                                color=color, linewidth=2, alpha=0.7)
                
                # Add markers at each gate change
                for point in change_points:
                    ax_progress.scatter(point, progress_data[point, i], color=color, 
                                       s=50, zorder=10, edgecolors='white', linewidths=1)
            
            # Add vertical line for current step
            ax_progress.axvline(x=len(steps)-1, color='#333333', linestyle='--', alpha=0.7, linewidth=1.5)
        except Exception as e:
            print(f"Error plotting progress: {e}")
    
    # Update altitude plot with enhanced styling
    if altitude_history:
        try:
            for i, agent in enumerate(env.agents):
                color = drone_colors[i % len(drone_colors)]
                if agent in altitude_history and len(altitude_history[agent]) > 0:
                    alt_data = altitude_history[agent]
                    steps = range(len(alt_data))
                    
                    # Plot altitude with gradient fill for better visibility
                    ax_altitude.plot(steps, alt_data, color=color, linewidth=2, alpha=0.8)
                    ax_altitude.fill_between(steps, 0, alt_data, color=color, alpha=0.1)
            
            # Add vertical line for current step
            ax_altitude.axvline(x=current_step, color='#333333', linestyle='--', alpha=0.7, linewidth=1.5)
        except Exception as e:
            print(f"Error plotting altitude: {e}")
    
    # Update speed plot
    if speed_history:
        try:
            for i, agent in enumerate(env.agents):
                color = drone_colors[i % len(drone_colors)]
                if agent in speed_history and len(speed_history[agent]) > 0:
                    speed_data = speed_history[agent]
                    steps = range(len(speed_data))
                    
                    # Plot speed with gradient fill for better visibility
                    ax_speed.plot(steps, speed_data, color=color, linewidth=2, alpha=0.8)
                    ax_speed.fill_between(steps, 0, speed_data, color=color, alpha=0.1)
            
            # Add vertical line for current step
            ax_speed.axvline(x=current_step, color='#333333', linestyle='--', alpha=0.7, linewidth=1.5)
        except Exception as e:
            print(f"Error plotting speed: {e}")
    
    # Add legend to progress plot with better styling
    handles = []
    labels = []
    for i, agent in enumerate(env.agents):
        color = drone_colors[i % len(drone_colors)]
        handles.append(plt.Line2D([0], [0], color=color, lw=2))
        labels.append(f'Drone {i}')
    
    # Create legend with custom styling
    legend = ax_progress.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.7,
                              fancybox=True, frameon=True, edgecolor='#cccccc')
    
    # Adjust tick marks to match gate indices
    ax_progress.set_yticks(range(len(env.gates)))
    ax_progress.set_yticklabels([f'G{i}' for i in range(len(env.gates))])
    
    # Create a professional status panel
    ax_status.clear()
    ax_status.axis('off')
    
    # Simulation title and header
    # Create header background
    header_rect = plt.Rectangle((0, 0.90), 1, 0.1, facecolor='#303030', alpha=0.8, transform=ax_status.transAxes)
    ax_status.add_patch(header_rect)
    
    # Add title
    ax_status.text(0.5, 0.95, "SIMULATION STATUS", fontsize=12, weight='bold', color='white',
                  ha='center', va='center', transform=ax_status.transAxes)
    
    # Simulation info with better styling
    sim_time = current_step * env.dt if hasattr(env, 'dt') else current_step * 0.1  # Default to 0.1s if dt not available
    
    # Create info table background
    info_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.85, facecolor='#f0f0f0', alpha=0.2, 
                             transform=ax_status.transAxes, edgecolor='#cccccc', linewidth=1)
    ax_status.add_patch(info_rect)
    
    # Add simulation info
    ax_status.text(0.1, 0.85, "Time:", fontsize=10, weight='bold')
    ax_status.text(0.4, 0.85, f"{sim_time:.1f}s", fontsize=10)
    
    ax_status.text(0.1, 0.80, "Step:", fontsize=10, weight='bold')
    ax_status.text(0.4, 0.80, f"{current_step}", fontsize=10)
    
    ax_status.text(0.1, 0.75, "Reward:", fontsize=10, weight='bold')
    ax_status.text(0.4, 0.75, f"{total_reward:.2f}", fontsize=10)
    
    collision_count = len(collided_drones)
    ax_status.text(0.1, 0.70, "Collisions:", fontsize=10, weight='bold')
    ax_status.text(0.4, 0.70, f"{collision_count}", fontsize=10, 
                  color='red' if collision_count > 0 else 'green')
    
    # Add separator line
    ax_status.axhline(y=0.65, xmin=0.08, xmax=0.92, color='#cccccc', alpha=0.5, linewidth=1)
    
    # Drone info table
    if len(positions_history) > 0:
        positions = positions_history[-1]
        velocities = velocities_history[-1]
        
        ax_status.text(0.1, 0.60, "DRONE STATUS", fontsize=10, weight='bold')
        
        y_offset = 0.55
        for i, agent in enumerate(env.agents):
            try:
                color = drone_colors[i % len(drone_colors)]
                pos = positions[i]
                vel = velocities[i]
                speed = np.linalg.norm(vel)
                
                if len(gate_indices_history) > 0:
                    gate_idx = gate_indices_history[-1][i]
                else:
                    gate_idx = env.vehicles[agent]["current_target_index"]
                
                # Make sure gate index is within bounds
                gate_idx = min(gate_idx, len(env.gates) - 1)
                
                # Get gate position
                gate = env.gates[gate_idx]
                if isinstance(gate, dict):
                    gate_pos = gate["position"] if "position" in gate else gate["center"]
                else:
                    gate_pos = gate.position if hasattr(gate, "position") else gate.center
                
                dist_to_gate = np.linalg.norm(np.array(pos) - np.array(gate_pos))
                
                # Create drone status indicator background
                status_color = 'red' if i in collided_drones else '#E8F5E9'
                drone_rect = plt.Rectangle((0.08, y_offset-0.13), 0.84, 0.15, 
                                         facecolor=status_color, alpha=0.2, 
                                         transform=ax_status.transAxes, edgecolor=color, linewidth=1)
                ax_status.add_patch(drone_rect)
                
                # Add drone header
                header_color = 'white' if i in collided_drones else 'black'
                ax_status.text(0.12, y_offset, f"Drone {i}", fontsize=9, weight='bold', 
                              color=color)
                
                # Create status indicators
                if i in collided_drones:
                    status_text = "COLLISION"
                    status_color = 'red'
                else:
                    status_text = "OPERATIONAL"
                    status_color = 'green'
                
                ax_status.text(0.7, y_offset, status_text, fontsize=8, 
                              color=status_color, weight='bold')
                
                # Add telemetry info
                ax_status.text(0.15, y_offset-0.04, f"Target: Gate {gate_idx}", fontsize=8)
                ax_status.text(0.15, y_offset-0.08, f"Dist: {dist_to_gate:.2f}m", fontsize=8)
                ax_status.text(0.6, y_offset-0.04, f"Speed: {speed:.2f}m/s", fontsize=8)
                ax_status.text(0.6, y_offset-0.08, f"Alt: {pos[2]:.2f}m", fontsize=8)
                
                y_offset -= 0.17
            except Exception as e:
                print(f"Error updating status for drone {i}: {e}")
    
    # Add simulation info and metrics in the info panel
    ax_info.clear()
    ax_info.axis('off')
    
    # Create header background
    header_rect = plt.Rectangle((0, 0.90), 1, 0.1, facecolor='#303030', alpha=0.8, transform=ax_info.transAxes)
    ax_info.add_patch(header_rect)
    
    # Add title
    ax_info.text(0.5, 0.95, "SIMULATION METRICS", fontsize=12, weight='bold', color='white',
                ha='center', va='center', transform=ax_info.transAxes)
    
    # Create info background
    info_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.85, facecolor='#f0f0f0', alpha=0.2, 
                             transform=ax_info.transAxes, edgecolor='#cccccc', linewidth=1)
    ax_info.add_patch(info_rect)
    
    # Add metrics
    # Calculate current metrics
    current_speeds = []
    current_target_dists = []
    collisions = 0
    completed_gates = []
    
    if len(positions_history) > 0:
        positions = positions_history[-1]
        velocities = velocities_history[-1]
        
        for i, agent in enumerate(env.agents):
            # Speed
            speed = np.linalg.norm(velocities[i])
            current_speeds.append(speed)
            
            # Distance to target
            if len(gate_indices_history) > 0:
                gate_idx = gate_indices_history[-1][i]
            else:
                gate_idx = env.vehicles[agent]["current_target_index"]
            
            gate_idx = min(gate_idx, len(env.gates) - 1)
            gate = env.gates[gate_idx]
            
            if isinstance(gate, dict):
                gate_pos = gate["position"] if "position" in gate else gate["center"]
            else:
                gate_pos = gate.position if hasattr(gate, "position") else gate.center
            
            dist = np.linalg.norm(np.array(positions[i]) - np.array(gate_pos))
            current_target_dists.append(dist)
            
            # Completed gates
            completed_gates.append(gate_idx)
            
            # Collisions
            if i in collided_drones:
                collisions += 1
    
    # Calculate statistics
    avg_speed = np.mean(current_speeds) if current_speeds else 0
    min_dist = np.min(current_target_dists) if current_target_dists else float('inf')
    max_dist = np.max(current_target_dists) if current_target_dists else 0
    avg_progress = np.mean(completed_gates) if completed_gates else 0
    
    # Display metrics
    ax_info.text(0.1, 0.85, "Current Status", fontsize=10, weight='bold')
    
    ax_info.text(0.1, 0.78, "Average Speed:", fontsize=9)
    ax_info.text(0.5, 0.78, f"{avg_speed:.2f} m/s", fontsize=9)
    
    ax_info.text(0.1, 0.73, "Closest Gate:", fontsize=9)
    ax_info.text(0.5, 0.73, f"{min_dist:.2f} m", fontsize=9)
    
    ax_info.text(0.1, 0.68, "Furthest Gate:", fontsize=9)
    ax_info.text(0.5, 0.68, f"{max_dist:.2f} m", fontsize=9)
    
    ax_info.text(0.1, 0.63, "Avg Progress:", fontsize=9)
    ax_info.text(0.5, 0.63, f"Gate {avg_progress:.1f}", fontsize=9)
    
    ax_info.text(0.1, 0.58, "Active Collisions:", fontsize=9)
    ax_info.text(0.5, 0.58, f"{collisions}", fontsize=9, color='red' if collisions > 0 else 'green')
    
    # Add separator
    ax_info.axhline(y=0.54, xmin=0.08, xmax=0.92, color='#cccccc', alpha=0.5, linewidth=1)
    
    # Add guidance/key information
    ax_info.text(0.1, 0.48, "Legend & Information", fontsize=10, weight='bold')
    
    # Add color-coded drone info
    y_pos = 0.43
    for i, agent in enumerate(env.agents[:min(4, len(env.agents))]):  # Show up to 4 drones to avoid clutter
        color = drone_colors[i % len(drone_colors)]
        ax_info.plot([0.1, 0.2], [y_pos, y_pos], color=color, linewidth=2)
        ax_info.text(0.22, y_pos-0.01, f"Drone {i}", fontsize=8)
        y_pos -= 0.05
    
    # Add gate info
    y_pos = 0.43
    for i in range(min(3, len(env.gates))):  # Show limited gate info to avoid clutter
        color = gate_colors[i]
        ax_info.plot([0.5, 0.6], [y_pos, y_pos], color=color, linewidth=2)
        ax_info.text(0.62, y_pos-0.01, f"Gate {i}", fontsize=8)
        y_pos -= 0.05
    
    # Add collision warning info
    collision_marker = plt.Circle((0.15, 0.25), 0.02, color='black', fill=True)
    ax_info.add_patch(collision_marker)
    ax_info.text(0.22, 0.24, "Normal Drone", fontsize=8)
    
    collision_marker = plt.Circle((0.15, 0.20), 0.02, color='red', fill=True)
    ax_info.add_patch(collision_marker)
    ax_info.text(0.22, 0.19, "Collision Warning", fontsize=8)
    
    # Add copyright/author info
    ax_info.text(0.5, 0.05, "Multi-Drone Reinforcement Learning Simulation", 
                fontsize=8, style='italic', ha='center')
    
    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax_info.text(0.5, 0.02, f"Generated: {timestamp}", fontsize=7, alpha=0.7, ha='center')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    return fig



def create_video_from_frames_v2(frame_folder, output_video_path, fps=15):
    """
    Creates a video from a sequence of image frames using OpenCV for better compatibility.
    
    Args:
        frame_folder: Folder containing the frame images
        output_video_path: Path where the video will be saved
        fps: Frames per second for the video
    """
    try:
        import cv2
        import os
        import glob
        
        # Sort frames to ensure proper sequence
        frame_files = sorted(glob.glob(os.path.join(frame_folder, "step_*.png")))
        
        if not frame_files:
            print(f"No frame files found in {frame_folder}")
            return
        
        print(f"Creating video from {len(frame_files)} frames...")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Add each frame to the video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_file}")
        
        # Release the video writer
        video.release()
        
        print(f"Video created successfully: {output_video_path}")
        return output_video_path
    
    except ImportError:
        print("Error: OpenCV (cv2) is required to create videos.")
        print("Attempting to install it automatically...")
        
        try:
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            
            # Retry after installation
            return create_video_from_frames_v2(frame_folder, output_video_path, fps)
        except Exception as install_error:
            print(f"Could not install OpenCV: {install_error}")
            print("Please install it manually with: pip install opencv-python")
            return None
    
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        
        # Fallback message
        print("Video creation failed. The individual frames are still available in the output folder.")
        return None
    

def parse_env_parameters_from_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        text = file.read()
    
    parameters = {}
    
    # Split by lines and process each line
    lines = text.strip().split('\n')
    
    # Skip the first line which is "Training parameters:"
    for line in lines[1:]:
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # If we've reached the stage info, stop reading further.
            if key.lower() == 'env_name' or key.lower() == 'stage':
                break
            
            # Convert values to appropriate types
            if value.lower() == 'true':
                parameters[key] = True
            elif value.lower() == 'false':
                parameters[key] = False
            elif value.replace('.', '', 1).isdigit():
                # Check if it's a float or int
                if '.' in value:
                    parameters[key] = float(value)
                else:
                    parameters[key] = int(value)
            else:
                parameters[key] = value
    
    return parameters



def create_animation(episode_number, save_path='./frames', output_format='gif', 
                    output_path=None, fps=10, quality=95, resize_factor=1.0,
                    include_every_nth_frame=1):
    """
    Create a GIF or MP4 animation from saved episode frames.
    
    Parameters:
    - episode_number: The episode number to create the animation for (1-based indexing)
    - save_path: Directory where frames are saved
    - output_format: 'gif' or 'mp4'
    - output_path: Path for the output file (if None, saves in the save_path directory)
    - fps: Frames per second
    - quality: Quality of the output (0-100 for GIF, bitrate for MP4 if specified as string like '4M')
    - resize_factor: Factor to resize frames (1.0 = original size)
    - include_every_nth_frame: Include only every Nth frame to reduce file size
    
    Returns:
    - Path to the created animation file
    """
    # Create output path if not specified
    if output_path is None:
        output_path = os.path.join(save_path, f'gifs/episode_{episode_number}_animation.{output_format}')
    
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    # Find all frames for the specified episode
    pattern = os.path.join(save_path, f'episode_{episode_number}_step_*.png')
    frame_files = sorted(glob.glob(pattern))
    
    if not frame_files:
        print(f"No frames found for episode {episode_number} in {save_path}")
        return None
    
    # Keep only every Nth frame if specified
    if include_every_nth_frame > 1:
        frame_files = frame_files[::include_every_nth_frame]
    
    print(f"Processing {len(frame_files)} frames for episode {episode_number}")
    
    if output_format.lower() == 'gif':
        # Calculate frame duration in milliseconds
        frame_duration = int(1000 / fps)
        
        # Load all images and resize if needed
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            if resize_factor != 1.0:
                new_width = int(img.width * resize_factor)
                new_height = int(img.height * resize_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            images.append(img.copy())
        
        # Save as GIF
        print(f"Creating GIF with {len(images)} frames at {fps} FPS...")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=0,  # 0 = infinite loop
            optimize=True,
            quality=quality
        )
        
    elif output_format.lower() == 'mp4':
        # Use ffmpeg directly for better quality control
        try:
            # Create a list file for ffmpeg
            list_file = os.path.join(save_path, f'episode_{episode_number}_frames.txt')
            with open(list_file, 'w') as f:
                for frame_file in frame_files:
                    f.write(f"file '{os.path.abspath(frame_file)}'\n")
                    f.write(f"duration {1/fps}\n")
                # Add the last frame again to avoid 0 duration for last frame
                f.write(f"file '{os.path.abspath(frame_files[-1])}'\n")
            
            # Build ffmpeg command
            bitrate = quality if isinstance(quality, str) else f"{quality//10}M"
            cmd = [
                'ffmpeg', '-y',  # Force overwrite
                '-f', 'concat',  # Use concat demuxer
                '-safe', '0',    # Don't require absolute paths
                '-i', list_file, # Input from list file
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'slow',   # Better compression
                '-crf', '18',        # High quality
                '-pix_fmt', 'yuv420p',  # Standard pixel format
                '-vf', f'scale=trunc(iw*{resize_factor}):trunc(ih*{resize_factor})',  # Resize if needed
                output_path
            ]
            
            print("Running ffmpeg with command:", ' '.join(cmd))
            subprocess.run(cmd, check=True)
            
            # Clean up list file
            os.remove(list_file)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error creating MP4: {e}")
            print("Falling back to imageio...")
            
            # Fallback to imageio if ffmpeg command fails
            frames = []
            for frame_file in frame_files:
                img = imageio.imread(frame_file)
                if resize_factor != 1.0:
                    from skimage.transform import resize
                    new_shape = (int(img.shape[0] * resize_factor), 
                                int(img.shape[1] * resize_factor), 
                                img.shape[2])
                    img = resize(img, new_shape, preserve_range=True).astype(np.uint8)
                frames.append(img)
            
            imageio.mimsave(output_path, frames, fps=fps)
    
    else:
        print(f"Unsupported output format: {output_format}. Use 'gif' or 'mp4'.")
        return None
    
    print(f"Animation saved to: {output_path}")
    return output_path