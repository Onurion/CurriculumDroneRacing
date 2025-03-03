import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DroneNavEnv(gym.Env):
    """
    A simplified drone navigation environment.
    
    Dynamics:
      - acceleration = action (in 3 dimensions, each in [-1,1])
      - velocity = velocity + acceleration * dt, where dt = 0.1
      - position = position + velocity * dt
      
    The drone starts at position [0, 0, 0] with zero velocity.
    The mission is to reach target [2, 2, 3] with low velocity.
    Termination conditions:
      - Success: drone is within 0.2 m of the target and its velocity norm < 0.2.
      - Failure: drone is > 10 m from the target.
      - Timeout: 500 time steps.
      
    Observations (all normalized):
      1. Relative position to the target: (target - position)/10  (3 values)
      2. Normalized Euclidean distance: distance/10  (1 value)
      3. Velocity vector normalized by max_vel (assumed max_vel=10): velocity/10  (3 values)
      4. Velocity toward target (projection normalized by max_vel): (np.dot(velocity, unit_rel_pos))/10  (1 value)
      5. Direction to target (unit vector): unit_rel_pos (3 values)
      
    Reward:
      A linear continuous reward function scales the reward based on distance:
      reward = 1 - 2*(distance / 10) so that reward = 1 at the target and -1 when distance = 10.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(DroneNavEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 500
        self.current_step = 0
        
        # Define action space: acceleration in 3 dimensions in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation components:
        # 3 (relative position) + 1 (norm distance) + 3 (velocity) + 1 (velocity toward target) + 3 (direction to target) = 11
        # All values will be normalized between -1 and 1 (or 0 and 1 for the norm distance)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        
        # Environment constants
        self.target = np.array([2.0, 2.0, 3.0], dtype=np.float32)
        self.max_distance = 10.0  # Beyond this, the episode terminates.
        self.max_vel = 10.0       # Used to normalize velocity. (Assumed maximum expected velocity)
        
        # Initialize state
        self.position = None
        self.velocity = None

    def reset(self, seed=None):
        """Reset the environment state."""
        self.position = np.zeros(3, dtype=np.float32)  # starting at [0, 0, 0]
        self.velocity = np.zeros(3, dtype=np.float32)  # starting with 0 velocity
        self.current_step = 0
        return self._get_obs(), {}
    
    def step(self, action):
        """Apply action (acceleration) and update state."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = 0.5*action  # Limit the maximum acceleration to 0.5
        self.current_step += 1
        
        # Update dynamics
        acceleration = action
        self.velocity = self.velocity + acceleration * self.dt
        self.position = self.position + self.velocity * self.dt
        
        # Calculate relative position and distance to target
        rel_pos = self.target - self.position
        distance = np.linalg.norm(rel_pos)
        
        # Compute unit vector in the direction of the target (if distance > 0)
        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)
        
        # Compute the projected velocity toward the target
        vel_toward_target = np.dot(self.velocity, unit_rel_pos)
        
        # Compute reward: Linear scaling between 1 (at target) and -1 (at max_distance)
        reward = 1.0 - 2.0 * (distance / self.max_distance)
        reward = np.clip(reward, -1.0, 1.0)
        
        truncated = False
        # Check termination conditions
        done = False
        # Successful mission if close enough and low velocity.
        if distance < 0.2 and np.linalg.norm(self.velocity) < 0.2:
            done = True
            reward = 1.0  # success reward
        # If too far away from target, episode fails.
        elif distance > self.max_distance:
            done = True
        # Timeout termination.
        elif self.current_step >= self.max_steps:
            done = True
            truncated = True
        
        info = {}
        return self._get_obs(), reward, done, truncated, info
    
    def _get_obs(self):
        """Compute the normalized observation."""
        # Relative position normalized by max_distance
        rel_pos = self.target - self.position
        norm_rel_pos = rel_pos / self.max_distance  # in [-1,1] if within max_distance
        
        distance = np.linalg.norm(rel_pos)
        norm_distance = np.array([distance / self.max_distance], dtype=np.float32)
        
        # Normalized velocity vector (using max_vel)
        norm_velocity = self.velocity / self.max_vel
        
        # Compute unit direction toward target
        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)
        
        # Velocity toward target (projected velocity), normalized by max_vel
        vel_toward_target = np.dot(self.velocity, unit_rel_pos) / self.max_vel
        vel_toward_target = np.array([vel_toward_target], dtype=np.float32)
        
        # Concatenate observations into a single vector (total dim = 11)
        obs = np.concatenate([norm_rel_pos, norm_distance, norm_velocity, vel_toward_target, unit_rel_pos])
        # In case of numerical issues, clip to observation space bounds.
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
    
    def render(self, mode='human'):
        """Simple text-based rendering."""
        rel_pos = self.target - self.position
        distance = np.linalg.norm(rel_pos)
        obs = self._get_obs()
        print(f"Step: {self.current_step}, Position: {self.position}, Velocity: {self.velocity}, Distance: {distance:.3f}")
        # print ("obs: ", obs)
    
    def close(self):
        pass