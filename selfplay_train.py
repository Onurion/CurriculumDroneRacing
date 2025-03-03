from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_race_selfplay import *
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime


# =============================================================================
# Define Evaluation and Frozen Model Update Parameters
# =============================================================================
total_timesteps = 2e7         # Total number of training timesteps.
eval_freq = 5000                  # Evaluation frequency (in timesteps).
eval_episodes = 10                # Number of episodes to use in each evaluation.
win_rate_threshold = 0.6          # Win rate to trigger frozen model update.
n_gates = 5                       # Number of gates in the environment.
radius = 10.0
distance_exp_decay = 2.0
action_coeff= 2.82
distance_exp_decay= 2.00
w_distance= 0.52
w_distance_change= 1.51
w_deviation= 0.17
w_inactivity= 0.38
reward_type= 2
algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
date_str = datetime.now().strftime("%d%B_%H%M")
main_folder = f"{date_str}_selfplay_2drones"
env_args = {"n_gates": n_gates, "radius":radius, "action_coeff":action_coeff, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
            "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type}



def write_env_parameters(main_folder, env_args):
    params_file = os.path.join(main_folder, "parameters.txt")
    with open(params_file, "a") as f:
        f.write(f"Training parameters:\n")
        f.write(f"algorithm= {algorithm}\n")
        f.write(f"action_coeff= {env_args['action_coeff']:.2f}\n")
        f.write(f"n_gates= {env_args['n_gates']}\n")
        f.write(f"radius= {env_args['radius']:.2f}\n")
        f.write(f"distance_exp_decay= {env_args['distance_exp_decay']:.2f}\n")
        f.write(f"w_distance= {env_args['w_distance']:.2f}\n")
        f.write(f"w_distance_change= {env_args['w_distance_change']:.2f}\n")
        f.write(f"w_deviation= {env_args['w_deviation']:.2f}\n")
        f.write(f"w_inactivity= {env_args['w_inactivity']:.2f}\n")
        f.write(f"reward_type= {env_args["reward_type"]}\n")
        f.write(f"total_timesteps= {total_timesteps}\n")
        f.write(f"eval_freq= {eval_freq}\n")
        f.write(f"eval_episodes= {eval_episodes}\n")
        f.write(f"win_rate_threshold= {win_rate_threshold}\n\n")

def make_env():
    base_env = DroneRaceSelfPlayEnv(**env_args)
    dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
    env = SelfPlayWrapper(base_env, dummy_opponent_policy)
    monitor_file = os.path.join(main_folder, "train_monitor.csv")
    env = Monitor(env, filename=monitor_file, allow_early_resets=True)
    return env

def make_eval_env():
    base_env = DroneRaceSelfPlayEnv(**env_args)
    dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
    env = SelfPlayWrapper(base_env, dummy_opponent_policy)
    monitor_file = os.path.join(main_folder, "eval_monitor.csv")
    env = Monitor(env, filename=monitor_file, allow_early_resets=True)
    return env

num_envs = 4  # Number of parallel environments

# Create vectorized environments for training and frozen model.
vec_train_env = DummyVecEnv([make_env for _ in range(num_envs)])
vec_frozen_env = DummyVecEnv([make_env for _ in range(num_envs)])

write_env_parameters(main_folder, env_args)

# Initialize the training model using the vectorized training environment.
# Initialize the frozen model using the vectorized frozen environment and copy initial parameters.
if algorithm == "ppo":
    model = PPO("MlpPolicy", vec_train_env, tensorboard_log= main_folder + "/tensorboard/", device="cpu", verbose=1)
    frozen_model = PPO("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
elif algorithm == "sac":
    model = SAC("MlpPolicy", vec_train_env, tensorboard_log= main_folder + "/tensorboard/", device="cpu", verbose=1)
    frozen_model = SAC("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)


frozen_model.set_parameters(model.get_parameters())

# Create the frozen opponent policy using the frozen model.
frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)

# Update each sub-environment in the vectorized training env to use the frozen opponent policy.
for env in vec_train_env.envs:
    env.frozen_opponent_policy = frozen_opponent_policy

# Create a separate single-instance evaluation environment.
eval_env = make_eval_env()
eval_env.frozen_opponent_policy = frozen_opponent_policy  # Ensure it uses the same frozen policy.

# =============================================================================
# Create Callbacks:
# 1. EvalCallback (built-in) for saving the best model based on reward.
# 2. FrozenModelUpdateCallback (custom) for updating the frozen opponent.
# =============================================================================

# Built-in EvalCallback for saving the best model during training.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path= main_folder + "/best_model/",
    log_path= main_folder + "/logs",
    eval_freq=eval_freq,
    deterministic=True,
    render=False,
)

# Custom callback for self-play frozen model update based on win rate.
frozen_update_callback = FrozenModelUpdateCallback(
    eval_env=eval_env,
    frozen_model=frozen_model,
    eval_freq=eval_freq,
    eval_episodes=eval_episodes,
    win_rate_threshold=win_rate_threshold,
    verbose=1,
)

# Combine both callbacks into a CallbackList.
callback_list = CallbackList([eval_callback, frozen_update_callback])


model.learn(total_timesteps=total_timesteps, callback=callback_list)


# Create a separate evaluation environment (non-vectorized)
# eval_env = make_env()  # Using the same make_env() as before
# obs = eval_env.reset()
# for episode in range(3):
#     done = False
#     print(f"\n--- Evaluation Episode: {episode} ---")
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = eval_env.step(action)
#         print(f"Reward: {reward}")
#         eval_env.render()
#     obs = eval_env.reset()





# # Create the base multi-agent self-play environment.
# base_env = DroneRaceSelfPlayEnv(n_gates=5)

# # For environment wrappers used in PPO initialization, we need a single-agent view.
# # Here we create a dummy opponent policy that returns zeros (will be overridden shortly).
# dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())

# # Create two instances of the SelfPlayWrapper:
# # One for training and one for the frozen model.
# train_env_wrapper = SelfPlayWrapper(base_env, dummy_opponent_policy)
# frozen_env_wrapper = SelfPlayWrapper(base_env, dummy_opponent_policy)

# # Initialize the training model.
# model = PPO("MlpPolicy", train_env_wrapper, device="cpu", verbose=1)

# # Create the frozen model (as a separate PPO instance) and copy initial weights.
# frozen_model = PPO("MlpPolicy", frozen_env_wrapper, device="cpu", verbose=0)
# frozen_model.set_parameters(model.get_parameters())

# # Initialize frozen opponent policy using the frozen model.
# frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
# # Now update the training wrapper to use the frozen policy.
# train_env_wrapper.frozen_opponent_policy = frozen_opponent_policy




# ----------------------------------------
# Evaluation: Run a few episodes to visualize performance.
# ----------------------------------------
# obs = train_env_wrapper.reset()
# for episode in range(3):
#     done = False
#     print(f"\n--- Evaluation Episode: {episode} ---")
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = train_env_wrapper.step(action)
#         print(f"Reward: {reward}")
#         train_env_wrapper.render()