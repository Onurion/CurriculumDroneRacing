from utils import *
from envs.drone_race_curriculum_multi_v8 import *
from envs.drone_race_curriculum_v7 import *
from stable_baselines3 import PPO, SAC
import time
import numpy as np
import pandas as pd
import os
from test_multiple import *
from utils import *


folder_path = "Results_26Feb_2025/26February_1146_curriculum_2drones_v3_stage_5"
selfplay = True
random_init = False
env_class = DroneRaceCurriculumEnv

algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
# env_args = {"n_gates": n_gates, "radius":radius, "action_coeff":action_coeff, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
#             "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type}

verbose = False
save_csv = True
n_eval_episodes = 10
n_episodes = 1000
jitter_range = [1.0, 2.0]

def evaluate_agent(env, model, n_episodes = 1000):
    # Optional testing loop.
    for i in range(n_eval_episodes):
        print("\nEvaluating agent episode: ", i)

        obs, info = env.reset()
        for _ in range(n_episodes):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print (f"Reward: {reward:.3f}")
            time.sleep(0.3)
            if done:
                print("Episode terminated!")
                break
        env.close()


#Evaluation results: Mean Reward=218.07, Mean Speed=0.49, Mean Targets Reached=33.10
#Evaluation results: Mean Reward=835.74, Mean Speed=0.42, Mean Targets Reached=38.15

# 1000 episode evaluation
#Evaluation results for 17February_1654_selfplay_2drones: Mean Reward=1557.60, Mean Speed=2.00/±0.23, Mean Targets Reached=28.20/±4.70
#Evaluation results for 17February_1701_selfplay_2drones: Mean Reward=646.64, Mean Speed=1.74/±0.31, Mean Targets Reached=25.64/±5.93

if __name__ == "__main__":

    param_file = os.path.join(folder_path, "parameters.txt")
    if not os.path.exists(param_file):
        print("  No parameters.txt found; skipping folder.")
        exit()

    # Read environment parameters from the file.
    params = read_parameters(param_file)

    if "n_agents" in params:
        n_agents = int(params["n_agents"])
    else:
        n_agents = 2

    # Convert and map parameters. For booleans do an explicit conversion.

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
        "gate_passing_tolerance": 0.5,  #float(params["gate_passing_tolerance"]),
        "drone_collision_margin": 0.25,  
        "takeover_reward": float(params["takeover_reward"]),
        "is_buffer_obs": is_buffer_obs,
        "buffer_size": buffer_size,
        "random_init": random_init,
        "jitter_range": jitter_range
    }
    if verbose:
        print("Environment parameters:", env_args)


    env = env_class(**env_args)
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


    # targets_reached_list, avg_speed_list, \
    # total_reward_list, collisions_list = evaluate_model(model, env, n_episodes=n_eval_episodes,
    #                                                         verbose=verbose, save_csv=save_csv,
    #                                                         main_folder=folder_path)

    import matplotlib
    matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

    import matplotlib.pyplot as plt

    save_path = "6March_2drones_v3_stage_5_results"


    # For evaluation with visualization
    targets, speeds, rewards, collisions = evaluate_model_with_visual(
        model, 
        env, 
        n_episodes=n_eval_episodes,
        max_steps=500,
        verbose=True,
        visualize=True,
        save_path=save_path,
        create_video=True
    )


    mean_reward = np.mean(rewards)
    std_reward = np.mean(rewards)
    mean_velocity = np.mean(speeds)
    std_velocity = np.std(speeds)
    mean_targets_reached = np.mean(targets)
    std_targets_reached = np.std(targets)
    mean_collision = np.mean(collisions)
    std_collision = np.std(collisions)

    print ("Evaluation results for", folder_path)
    print (f"Mean Reward={mean_reward:.2f}, Mean Speed={mean_velocity:.2f}/{std_velocity:.2f}, Mean Targets Reached={mean_targets_reached:.2f}/{std_targets_reached:.2f}, Mean Collision={mean_collision:.2f}/{std_collision:.2f}")

    results_txt_path = os.path.join(save_path, "results.txt")
    with open(results_txt_path, "a") as f:
        f.write(f"\nmean_targets_reached: {mean_targets_reached:.2f}/±{std_targets_reached:.2f}\n")
        f.write(f"mean_velocity: {mean_velocity:.2f}/±{std_velocity:.2f}\n")
        f.write(f"mean_collision: {mean_collision:.2f}/±{std_collision:.2f}\n")
        f.write(f"mean_reward: {mean_reward:.2f}/±{std_reward:.2f}\n")
    print("  Saved results.txt.")



    # if selfplay:
    #     dummy_opponent_policy = lambda obs: np.zeros_like(env.action_space["drone1"].sample())
    #     env = SelfPlayWrapper(env, dummy_opponent_policy)


    # model_path =  main_folder + "/best_model/best_model.zip"
    # model = PPO.load(model_path, device="cpu", env=env)  # Pass the env so that model.predict returns

    # if selfplay:
    #     frozen_model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    #     frozen_model.set_parameters(model.get_parameters())
    #     frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
    #     env.frozen_opponent_policy = frozen_opponent_policy

    # print ("Results for ", main_folder)
    # evaluation_df = evaluate_model(model, env, n_episodes=n_eval_episodes, verbose=verbose, save_csv=save_csv)

    # evaluate_agent(env, model, n_episodes)






