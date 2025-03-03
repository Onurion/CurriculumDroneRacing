import os
import optuna
import numpy as np

from stable_baselines3 import PPO, SAC
from drone_race_selfplay import *
from drone_race_curriculum import *
from curriculum_train import reinitialize_envs
from test_multiple import evaluate_model
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
from datetime import datetime

# -------------------------------
# Global configuration and variables
# -------------------------------

num_envs = 4
algorithm = "ppo"  # or "sac"
total_timesteps = int(1e6)  # For the sake of testing; use 1e7 in production runs
eval_freq = 2000
eval_episodes = 5
max_no_improvement_evals = 10
win_rate_threshold = 0.6
n_gates = 5
radius = 10.0
action_coefficient= 1.92
distance_exp_decay= 2.00
w_distance= 0.52
w_distance_change= 1.51
w_deviation= 0.17
w_inactivity= 0.20
reward_type= 1
observation_type= 1
is_buffer_obs= False
buffer_size= 5
max_steps = 2000
algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
date_str = datetime.now().strftime("%d%B_%H%M")
main_folder = f"{date_str}_flexible_curriculum"
env_args = {"n_gates": n_gates, "radius":radius, "action_coefficient":action_coefficient, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
            "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type, "observation_type": observation_type,
            "is_buffer_obs": is_buffer_obs, "buffer_size": buffer_size, "max_steps": max_steps}


# Global variables to hold the best model info across trials
best_model_path = None
best_score = -np.inf  # set as minus infinity to maximize properly

# For all trials
def log_trial_results(trial_number, score, mean_targets_reached, std_targets_reached,
                      mean_speed, std_speed, mean_collisions, std_collisions):
    log_line = (f"Trial {trial_number} Score: {score:.2f}, "
                f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

    with open(os.path.join(main_folder, "all_trials.txt"), "a") as f:
        f.write(log_line)


# --- Curriculum Scheduler Class ---
class CurriculumScheduler:
    def __init__(self, init_min_velocity=1.0, init_action_coefficient=1.0,
                 increase_step=0.5, decrease_step=0.3, success_threshold=25.0):
        """
        :param init_min_velocity: Initial minimum velocity.
        :param init_action_coefficient: Initial action coefficient.
        :param increase_step: The amount by which parameters are increased when performance is good.
        :param decrease_step: The amount by which parameters are decreased when performance is poor.
        :param success_threshold: A performance measure threshold (e.g., computed score) above which
                                  we consider training successful.
        """
        self.minimum_velocity = init_min_velocity
        self.action_coefficient = init_action_coefficient
        self.increase_step = increase_step
        self.decrease_step = decrease_step
        self.success_threshold = success_threshold
        self.difficulty_step = 1

    def update(self, score):
        """
        Update the difficulty parameters according to the recent performance score.
        Increase difficulty if performance is above the success threshold,
        otherwise decrease the difficulty.
        """
        if score >= self.success_threshold:
            print("Performance good: increasing difficulty.")
            self.minimum_velocity += self.increase_step
            self.action_coefficient += self.increase_step
        else:
            print("Performance poor: reducing difficulty a bit.")
            # Do not go below the starting point of 1.0
            self.minimum_velocity = max(1.0, self.minimum_velocity - self.decrease_step)
            self.action_coefficient = max(1.0, self.action_coefficient - self.decrease_step)

    def get_params(self):
        """
        Returns the current difficulty parameters.
        """
        stage_timesteps = 5e5 * self.difficulty_step
        env_args = {
            "minimum_velocity": self.minimum_velocity,
            "action_coefficient": self.action_coefficient
        }
        return env_args, stage_timesteps

# --- Main Adaptive Curriculum Training Function ---
def adaptive_curriculum_train(max_stages=10):
    """
    Adaptive training loop that adjusts difficulty based on performance.
    Uses transfer learning between stages.
    """
    # Create output folder if needed
    os.makedirs(main_folder, exist_ok=True)

    # Initialize curriculum scheduler with starting parameters (both 1.0)
    scheduler = CurriculumScheduler(init_min_velocity=1.0, init_action_coefficient=1.0,
                                    increase_step=1.0, decrease_step=0.3, success_threshold=0.5)

    # Placeholder for global best value and model
    best_model_path = None
    best_score = -np.inf

    # Loop over curriculum stages
    for stage in range(1, max_stages + 1):
        print(f"\n=== Starting Curriculum Stage {stage} ===")
        # Get current difficulty parameters from the scheduler.
        difficulty_params, stage_timesteps = scheduler.get_params()
        print(f"Current difficulty parameters: {difficulty_params}")

        # Combine these with other environment settings.
        env_params = {
            **difficulty_params,
            "enable_collision": True,
            "terminate_on_collision": False,
            "collision_penalty": 0.5,
            "gate_passing_tolerance": 0.25,
            "takeover_reward": 0.2,
        }

        current_env_args = dict(env_args)
        current_env_args.update(env_params)

        # Setup folder for this stage.
        stage_folder = os.path.join(main_folder, f"stage_{stage}")
        os.makedirs(stage_folder, exist_ok=True)
        write_env_parameters(stage_folder, current_env_args, stage=f"Stage_{stage}", algorithm=algorithm,
                             total_timesteps=stage_timesteps, eval_freq=eval_freq,
                             eval_episodes=eval_episodes, win_rate_threshold=win_rate_threshold)

        # --- (Re)initialize environments ---
        vec_train_env, vec_frozen_env, eval_env = reinitialize_envs(current_env_args, stage_folder, num_envs)

        # --- Transfer Learning: Load best model if available ---
        if best_model_path is not None and os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for stage {stage}")
            if algorithm == "ppo":
                model = PPO.load(best_model_path, env=vec_train_env, device="cpu",
                                 tensorboard_log=os.path.join(stage_folder, "tensorboard"))
            elif algorithm == "sac":
                model = SAC.load(best_model_path, env=vec_train_env, device="cpu",
                                 tensorboard_log=os.path.join(stage_folder, "tensorboard"))
            model.set_env(vec_train_env)
        else:
            if algorithm == "ppo":
                model = PPO("MlpPolicy", vec_train_env,
                            tensorboard_log=os.path.join(stage_folder, "tensorboard"),
                            device="cpu", verbose=1)
            elif algorithm == "sac":
                model = SAC("MlpPolicy", vec_train_env,
                            tensorboard_log=os.path.join(stage_folder, "tensorboard"),
                            device="cpu", verbose=1)

        # --- Setup frozen model for self-play ---
        if algorithm == "ppo":
            frozen_model = PPO("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
        elif algorithm == "sac":
            frozen_model = SAC("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
        frozen_model.set_parameters(model.get_parameters())

        # Configure the frozen opponent policy.
        frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
        for env in vec_train_env.envs:
            env.frozen_opponent_policy = frozen_opponent_policy
        eval_env.frozen_opponent_policy = frozen_opponent_policy

        # -------------------------------
        # Set up callbacks for evaluation and model updates.
        # -------------------------------
        improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=max_no_improvement_evals)

        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(stage_folder, "best_model"),
            log_path=os.path.join(stage_folder, "logs"),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=improvement_callback
        )
        
        frozen_update_callback = FrozenModelUpdateCallback(
            eval_env=eval_env,
            frozen_model=frozen_model,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            win_rate_threshold=win_rate_threshold,
            verbose=1,
        )
        callback_list = CallbackList([eval_callback, frozen_update_callback])
        
        # --- Train the model for the stage ---
        model.learn(total_timesteps=stage_timesteps, callback=callback_list)
        
        # --- Evaluate the trained model ---
        targets_reached_list, avg_speed_list, total_reward_list, collisions_list = \
                    evaluate_model(model, eval_env, n_episodes=eval_episodes)
        mean_reward = np.mean(total_reward_list)
        mean_speed = np.mean(avg_speed_list)
        std_speed = np.std(avg_speed_list)
        mean_targets_reached = np.mean(targets_reached_list)
        std_targets_reached = np.std(targets_reached_list)
        mean_collisions = np.mean(collisions_list)
        std_collisions = np.std(collisions_list)
        
        # Define a performance score that incorporates your key metrics.
        # You can adjust the score formula according to what matters most.
        score = (mean_targets_reached - std_targets_reached / 2.0) / 50.0 + \
                (mean_speed - std_speed / 2.0) / 10.0 - \
                (mean_collisions + std_collisions / 2.0)
        
        print(f"\nStage {stage} Evaluation Parameters: minimum_velocity {difficulty_params['minimum_velocity']:.2f}  action_coefficient {difficulty_params['action_coefficient']:.2f}")
        print(f"Results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_speed:.2f}/±{std_speed:.2f}, " 
              f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f}, Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")
        print(f"Stage {stage} finished with score: {score:.2f}")

        log_trial_results(stage, score, mean_targets_reached, std_targets_reached,
                          mean_speed, std_speed, mean_collisions, std_collisions)
        
        # --- Transfer Learning: Save the model if it improved the global best score ---
        if score > best_score:
            print(f"New best score achieved: {score:.2f} (old best: {best_score:.2f}). Saving new best model.")

            # Log the new best score
            log_line = (f"Stage {stage} Score: {score:.2f}, "
                        f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                        f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                        f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

            with open(os.path.join(main_folder, "best_scores.txt"), "a") as f:
                f.write(log_line)

            best_score = score
            best_model_folder = os.path.join(stage_folder, "best_model")
            os.makedirs(best_model_folder, exist_ok=True)
            best_model_path = os.path.join(best_model_folder, "best_model.zip")
            model.save(best_model_path)
        else:
            print(f"Stage {stage} did not beat the best score: {best_score:.2f}.")
            # Optionally reload the global best model for the next stage if desired.
        
        # --- Update curriculum difficulty based on performance ---
        scheduler.update(score)

    print("\nAdaptive curriculum training completed.")
    print(f"Best score achieved: {best_score:.2f}")
    return best_model_path

if __name__ == "__main__":
    adaptive_curriculum_train(max_stages=20)