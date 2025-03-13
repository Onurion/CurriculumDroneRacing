import os
import time
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from envs.drone_race_curriculum_multi_v8 import *
from envs.drone_race_centralized_v8 import *
from utils import *

root_dir = "Results_27Feb_2025"
n_eval_episodes = 10
random_init = False
terminate_on_collision = True
selfplay = True
env_class = DroneRaceCurriculumMultiEnv

def read_parameters(filepath):
    """
    Read parameters from a text file and return a dict.
    This function reads each line until it reaches the line that starts with 'stage='.
    """
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines or header lines
            if not line or line.startswith("Training parameters:"):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
                # If we've reached the stage info, stop reading further.
                if key.lower() == 'stage':
                    break
    return params

def evaluate_agent(env, model, n_eval_episodes=10, n_episodes=1000):
    # Optional testing loop.
    for i in range(n_eval_episodes):
        print("\nEvaluating agent episode:", i)
        obs, info = env.reset()
        for _ in range(n_episodes):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print(f"Reward: {reward:.3f}")
            time.sleep(0.3)
            if done:
                print("Episode terminated!")
                break
        env.close()

def evaluate_model(model, env, n_episodes=10, verbose=False, save_csv=False, main_folder=""):
    """
    Evaluate a given model in the provided environment.
    Returns a Pandas DataFrame with metrics aggregated over episodes.
    """
    episode_stats = []  # To store metrics per episode
    total_reward_list = []
    avg_speed_list = []
    targets_reached_list = []
    collisions_list = []

    base_env = get_base_env_with_agents(env)
    if not hasattr(base_env, "agents"):
        raise AttributeError("The base environment does not have an 'agents' attribute.")

    agents = base_env.agents

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        speeds = {}          # To store each agent's speeds over steps
        targets_reached = {}  # To store the final progress reported for each agent

        for agent in agents:
            speeds[agent] = []
            targets_reached[agent] = 0.0

        # Run one episode.
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            for agent in agents:
                state = base_env.get_state(agent)
                speeds[agent].append(np.linalg.norm(state[3:6]))
                targets_reached[agent] = base_env.vehicles[agent]["progress"]

        total_reward_list.append(total_reward)
        if verbose:
            print(f"Episode {episode}: Total Reward={total_reward:.2f}, Steps={steps}")

        avg_speed = {}
        for agent in agents:
            # Compute the average speed per agent.
            avg_speed[agent] = np.mean(speeds[agent])
            if verbose:
                print(f"Agent: {agent} Targets Reached={targets_reached[agent]}, Avg Speed={avg_speed[agent]:.2f}")
            avg_speed_list.append(avg_speed[agent])
            targets_reached_list.append(targets_reached[agent])
            collisions_list.append(info[agent]["collision"])


        episode_stats.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "targets_reached": targets_reached,
            "avg_speed": avg_speed,
            "collisions:": collisions_list
        })

    mean_reward = np.mean(total_reward_list)
    mean_speed = np.mean(avg_speed_list)
    std_speed = np.std(avg_speed_list)
    mean_targets_reached = np.mean(targets_reached_list)
    std_targets_reached = np.std(targets_reached_list)
    mean_collisions = np.mean(collisions_list)
    std_collisions = np.std(collisions_list)

    print(f"\nEvaluation results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_speed:.2f}/±{std_speed:.2f}, "
          f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")

    if save_csv:
        stats_df = pd.DataFrame(episode_stats)
        results_csv = os.path.join(main_folder, "evaluation_results.csv")
        stats_df.to_csv(results_csv, index=False)

    return targets_reached_list, avg_speed_list, total_reward_list, collisions_list



def str_to_bool(s):
    return s.strip().lower() in ["true", "1", "yes"]

if __name__ == "__main__":
    # Set the main folder where your parameters.txt and trained model are stored.

    

    verbose = False
    save_csv = True

    results_summary = {}

    # Iterate over each folder in root_dir.
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # skip non-folders

        print(f"\nProcessing folder: {folder_path}")
        # Path to parameters.txt in the folder.
        param_file = os.path.join(folder_path, "parameters.txt")
        if not os.path.exists(param_file):
            print("  No parameters.txt found; skipping folder.")
            continue

        # Read environment parameters from the file.
        params = read_parameters(param_file)

        # Convert and map parameters. For booleans do an explicit conversion.

        if "n_agents" in params:
            n_agents = int(params["n_agents"])
        else:
            n_agents = 2

        is_buffer_obs = str_to_bool(params.get("is_buffer_obs", "False"))
        buffer_size = int(params.get("buffer_size", 10))

        # Parse and convert parameters as needed.
        # Note: The file uses "action_coefficient" but your environment expects "action_coeff".
        env_args = {
            "n_agents": n_agents,
            "n_gates": int(params["n_gates"]),
            "radius": float(params["radius"]),
            "action_coefficient": float(params["action_coefficient"]),
            "distance_exp_decay": float(params["distance_exp_decay"]),
            "w_distance": float(params["w_distance"]),
            "w_distance_change": float(params["w_distance_change"]),
            "w_deviation": float(params["w_deviation"]),
            "w_inactivity": float(params["w_inactivity"]),
            "reward_type": int(params["reward_type"]),
            "observation_type": int(params["observation_type"]),
            "minimum_velocity": float(params["minimum_velocity"]),
            "enable_collision": bool(params["enable_collision"]),
            "terminate_on_collision": terminate_on_collision,
            "collision_penalty": float(params["collision_penalty"]),
            "gate_passing_tolerance": float(params["gate_passing_tolerance"]),
            "takeover_reward": float(params["takeover_reward"]),
            "is_buffer_obs": is_buffer_obs,
            "buffer_size": buffer_size,
            "random_init": random_init
        }
        if verbose:
            print("Environment parameters:", env_args)

        # Create the environment.
        try:
            env = env_class(**env_args)
        except Exception as e:
            print("  Error creating environment:", e)
            continue

        if selfplay:
            original_action_spaces = env.action_space  # This is still a Dict

            # First create dummy policies (these are just placeholders)
            dummy_opponent_policies = {}
            for i in range(1, n_agents):
                agent_id = f"drone{i}"
                dummy_opponent_policies[agent_id] = lambda obs: np.zeros_like(original_action_spaces[agent_id].sample())

            # Wrap the environment
            env = SelfPlayWrapper(env, dummy_opponent_policies)
        

        # Load the model
        model_path = os.path.join(folder_path, "best_model", "best_model.zip")
        model = PPO.load(model_path, device="cpu", env=env)

        if selfplay:
            # Create frozen model
            frozen_model = PPO("MlpPolicy", env, device="cpu", verbose=0)
            frozen_model.set_parameters(model.get_parameters())
            frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
            
            # Set the frozen policy for each opponent
            for agent_id in env.frozen_opponent_policies.keys():
                env.frozen_opponent_policies[agent_id] = frozen_opponent_policy


        targets_reached_list, avg_speed_list, \
        total_reward_list, collisions_list = evaluate_model(model, env, n_episodes=n_eval_episodes,
                                                              verbose=verbose, save_csv=save_csv,
                                                              main_folder=folder_path)


        mean_reward = np.mean(total_reward_list)
        mean_velocity = np.mean(avg_speed_list)
        std_velocity = np.std(avg_speed_list)
        mean_targets_reached = np.mean(targets_reached_list)
        std_targets_reached = np.std(targets_reached_list)
        mean_collision = np.mean(collisions_list)
        std_collision = np.std(collisions_list)

        # Create results.txt in the folder.
        results_txt_path = os.path.join(folder_path, "results.txt")
        with open(results_txt_path, "a") as f:
            f.write(f"mean_targets_reached: {mean_targets_reached:.2f}/±{std_targets_reached:.2f}\n")
            f.write(f"mean_velocity: {mean_velocity:.2f}/±{std_velocity:.2f}\n")
            f.write(f"mean_collision: {mean_collision:.2f}/±{std_collision:.2f}\n")
        print("  Saved results.txt.")

        # Store the results summary in the dictionary.
        results_summary[folder] = {
            "mean_target": mean_targets_reached,
            "std_target": std_targets_reached,
            "mean_velocity": mean_velocity,
            "std_velocity": std_velocity,
            "mean_collision": mean_collision
        }


    # Sort the folders by mean_targets_reached.
    sorted_results = sorted(results_summary.items(), key=lambda item: item[1]["mean_target"], reverse=True)

    print("\nSorted Results (by mean_target, descending):")
    for folder, metrics in sorted_results:
        print(f"{folder}: {metrics}")

    # Optionally, store the full sorted results in a global summary file.
    global_summary_path = os.path.join(root_dir, "global_results_summary_v2.txt")
    with open(global_summary_path, "a") as f:
        for folder, metrics in sorted_results:
            f.write(f"{folder}: {metrics}\n")
    print(f"Global summary saved in {global_summary_path}")
