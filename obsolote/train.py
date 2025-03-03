from envs.drone_vel_gate_env_v4 import DroneVelGateEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv  # or SubprocVecEnv for heavier envs
from stable_baselines3.common.monitor import Monitor
import os


n_gates = 5
main_folder = f"./9th_february_drone_vel_gate_{n_gates}_sac"

# Create separate functions for training and evaluation environments
def make_train_env():
    env = DroneVelGateEnv(n_gates=n_gates)  # Your custom environment
    # Write training monitor CSV to a specific file (for example, "train_monitor.csv")
    monitor_file = os.path.join(main_folder, "train_monitor.csv")
    return Monitor(env, filename=monitor_file, allow_early_resets=True)

def make_eval_env():
    env = DroneVelGateEnv(n_gates=n_gates)  # Your custom environment
    # Write evaluation monitor CSV to a different file (for example, "eval_monitor.csv")
    monitor_file = os.path.join(main_folder, "eval_monitor.csv")
    return Monitor(env, filename=monitor_file, allow_early_resets=True)

def train():
    # --- Create vectorized training environment ---
    n_train_envs = 4  # Number of environments in parallel for training.
    env = DummyVecEnv([make_train_env for _ in range(n_train_envs)])

    # --- Create vectorized evaluation environment ---
    # Often for evaluation a single environment (or a separate vectorized env) is used.
    n_eval_envs = 1
    eval_env = DummyVecEnv([make_eval_env for _ in range(n_eval_envs)])

    # Set up a callback for evaluation (optional).
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path= main_folder + '/best_model/',
        log_path= main_folder + '/results/',
        eval_freq=6000,
        deterministic=True,
        render=False
    )

    # Initialize PPO with the custom environment.
    model = SAC("MlpPolicy", env, device="cpu", tensorboard_log= main_folder + "/tensorboard/", verbose=1)

    # Train the agent.
    total_timesteps = 5e7  # Adjust as needed.
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)



if __name__ == "__main__":
    train()

    # Example usage:
    # env = MultiTargetDroneEnv()
    # obs = env.reset()
    # for _ in range(250):
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, _ = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()