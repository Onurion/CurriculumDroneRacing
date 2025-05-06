import os
import time
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from envs.drone_race_curriculum_multi import *
from utils import *

root_dir = "Results_nocurriculum_infinity" 
n_eval_episodes = 5
random_init = False
evaluation_mode = True
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
                # If we've reached the stage info, stop reading further.
                if key.lower() == 'env_name':
                    break

                value = value.strip()
                params[key] = value
                
    return params



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
    collision_arg = "collision"

    base_env = get_base_env_with_agents(env)
    if not hasattr(base_env, "agents"):
        raise AttributeError("The base environment does not have an 'agents' attribute.")

    agents = base_env.agents

    positions_list = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        speeds = {}          # To store each agent's speeds over steps
        targets_reached = {}  # To store the final progress reported for each agent
        positions = {}

        for agent in agents:
            positions[agent] = []
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
                positions[agent].append(state[:3])
                speeds[agent].append(np.linalg.norm(state[3:6]))
                targets_reached[agent] = base_env.vehicles[agent]["progress"]

        total_reward_list.append(total_reward)
        positions_list.append(positions)
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
            collisions_list.append(info[agent][collision_arg])


        episode_stats.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "targets_reached": targets_reached,
            "avg_speed": avg_speed,
            "collisions:": collisions_list
        })

    mean_reward = np.mean(total_reward_list)
    mean_velocity = np.mean(avg_speed_list)
    std_velocity = np.std(avg_speed_list)
    mean_collisions = np.mean(collisions_list)
    std_collisions = np.std(collisions_list)

    total_distance = 0
    gate_positions = env.env.gate_positions
    for i in range(len(gate_positions)):
        current_gate = gate_positions[i]
        next_gate = gate_positions[(i + 1) % len(gate_positions)]  # Wrap around for complete lap
        segment_distance = np.linalg.norm(next_gate - current_gate)
        total_distance += segment_distance
    
    # Calculate mean lap time
    mean_lap_time = total_distance / mean_velocity
    
    # For standard deviation of lap time, we use error propagation:
    # If T = D/V, then σ_T/T = σ_V/V (for constant distance)
    relative_std_velocity = std_velocity / mean_velocity
    std_lap_time = mean_lap_time * relative_std_velocity

    print(f"\nEvaluation results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_velocity:.2f}/±{std_velocity:.2f}, "
          f"Mean Lap Time={mean_lap_time:.2f}/±{std_lap_time:.2f} Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")

    if save_csv:
        stats_df = pd.DataFrame(episode_stats)
        results_csv = os.path.join(main_folder, "evaluation_results.csv")
        stats_df.to_csv(results_csv, index=False)

        # pickle the positions list
        with open(os.path.join(main_folder, "positions_list.pkl"), "wb") as f:
            pickle.dump(positions_list, f)


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

        # if not folder.startswith("19March"):
        #     continue  # Skip folders that don't start with "19_March"

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
        env_args = parse_env_parameters_from_file(param_file)

        # Convert and map parameters. For booleans do an explicit conversion.

        if "n_agents" in env_args:
            n_agents = int(env_args["n_agents"])
        else:
            n_agents = 2

        track_type = env_args.get("track_type", "circle")
        gate_size = float(env_args.get("gate_size", 2.0))


        # Parse and convert parameters as needed.
        # Note: The file uses "action_coefficient" but your environment expects "action_coeff".
        env_args.update({
            "n_agents": n_agents,
            "evaluation_mode": evaluation_mode,
            "terminate_on_collision": terminate_on_collision,
            "gate_passing_tolerance": 1.0,
            "drone_collision_margin": 0.25,
            "gate_size": gate_size,
            "track_type": track_type,
            "random_init": random_init
        })


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

        total_distance = 0
        gate_positions = env.env.gate_positions
        for i in range(len(gate_positions)):
            current_gate = gate_positions[i]
            next_gate = gate_positions[(i + 1) % len(gate_positions)]  # Wrap around for complete lap
            segment_distance = np.linalg.norm(next_gate - current_gate)
            total_distance += segment_distance
        
        # Calculate mean lap time
        mean_lap_time = total_distance / mean_velocity
        
        # For standard deviation of lap time, we use error propagation:
        # If T = D/V, then σ_T/T = σ_V/V (for constant distance)
        relative_std_velocity = std_velocity / mean_velocity
        std_lap_time = mean_lap_time * relative_std_velocity


        # Create results.txt in the folder.
        results_txt_path = os.path.join(folder_path, "results.txt")
        with open(results_txt_path, "a") as f:
            f.write(f"mean_lap_time: {mean_lap_time:.2f}/±{std_lap_time:.2f}\n")
            f.write(f"mean_velocity: {mean_velocity:.2f}/±{std_velocity:.2f}\n")
            f.write(f"mean_collision: {mean_collision:.2f}/±{std_collision:.2f}\n")
        print("  Saved results.txt.")

        # Store the results summary in the dictionary.
        results_summary[folder] = {
            "mean_lap_time": mean_lap_time,
            "std_lap_time": std_lap_time,
            "mean_velocity": mean_velocity,
            "std_velocity": std_velocity,
            "mean_collision": mean_collision
        }


    # Sort the folders by mean_lap_time.
    sorted_results = sorted(results_summary.items(), key=lambda item: item[1]["mean_velocity"], reverse=True)

    print("\nSorted Results (by mean_lap_time, descending):")
    for folder, metrics in sorted_results:
        print(f"{folder}: {metrics}")

    # Optionally, store the full sorted results in a global summary file.
    global_summary_path = os.path.join(root_dir, "global_results_summary.txt")
    with open(global_summary_path, "a") as f:
        for folder, metrics in sorted_results:
            f.write(f"{folder}: {metrics}\n")
    print(f"Global summary saved in {global_summary_path}")
