from utils import *
from envs.drone_race_curriculum_v7 import DroneRaceCurriculumEnv
from stable_baselines3 import PPO, SAC
import time
import numpy as np
import pandas as pd
import os
from test_multiple import *
from utils import *


main_folder = "Results_22Feb_2025/22February_1610_curriculum_2drones_v3_stage_5"
selfplay = True
random_init = False
env_class = DroneRaceCurriculumEnv

algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
# env_args = {"n_gates": n_gates, "radius":radius, "action_coeff":action_coeff, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
#             "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type}

verbose = False
save_csv = True
n_eval_episodes = 100
n_episodes = 1000

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

def get_base_env_with_agents(env):
    """
    Recursively unwrap the environment until an object containing the attribute 'agents' is found.
    """
    current_env = env
    while not hasattr(current_env, "agents") and hasattr(current_env, "env"):
        current_env = current_env.env
    return current_env

if __name__ == "__main__":

    param_file = os.path.join(main_folder, "parameters.txt")
    if not os.path.exists(param_file):
        print("  No parameters.txt found; skipping folder.")
        exit()

    # Read environment parameters from the file.
    params = read_parameters(param_file)

    # Convert and map parameters. For booleans do an explicit conversion.

    is_buffer_obs = str_to_bool(params.get("is_buffer_obs", "False"))
    buffer_size = int(params.get("buffer_size", 10))

    # Parse and convert parameters as needed.
    # Note: The file uses "action_coefficient" but your environment expects "action_coeff".
    env_args = {
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


    env = env_class(**env_args)
    if selfplay:
        dummy_opponent_policy = lambda obs: np.zeros_like(env.action_space["drone1"].sample())
        env = SelfPlayWrapper(env, dummy_opponent_policy)


    model_path =  main_folder + "/best_model/best_model.zip"
    model = PPO.load(model_path, device="cpu", env=env)  # Pass the env so that model.predict returns

    if selfplay:
        frozen_model = PPO("MlpPolicy", env, device="cpu", verbose=0)
        frozen_model.set_parameters(model.get_parameters())
        frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
        env.frozen_opponent_policy = frozen_opponent_policy

    print ("Results for ", main_folder)
    evaluation_df = evaluate_model(model, env, n_episodes=n_eval_episodes, verbose=verbose, save_csv=save_csv)

    # evaluate_agent(env, model, n_episodes)






