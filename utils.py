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