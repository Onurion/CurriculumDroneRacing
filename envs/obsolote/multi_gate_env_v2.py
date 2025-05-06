import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

class MultiGateEnv(gym.Env):
    """
    A cyclic drone navigation environment with 5 targets arranged in a circular trajectory.
    
    Dynamics:
      - Action: acceleration in 3 dimensions (each in [-1, 1])
      - State update: velocity and position are updated with a fixed dt.
    
    Task:
      - The drone must navigate between 5 targets in order.
      - On reaching the current target (within a threshold distance and low velocity),
        the target is updated (cyclically).
    
    Observations:
      - Relative position to the current target (normalized)
      - The norm of this relative position (normalized)
      - The drone’s current velocity (and its normalized value)
      - Optionally: a one-hot encoding representing the current target index.
      
    Rewards:
      - A base reward based on the current distance (closer is better).
      - A bonus when a target is reached.
      - Penalties for high velocity and misalignment, similar to previous structure.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MultiGateEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0
        self.gate_size = 1.0  # 1m x 1m gate
        self.passing_tolerance = self.gate_size / 2.0

        # Drone state: position and velocity in 3D.
        self.position = None
        self.velocity = None
        
        # Define 5 target positions arranged in a circle (for simplicity, in the XY plane with different altitudes)
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 evenly spaced angles
        radius = 5.0
        
        # Create gates; each gate is a 1m x 1m rectangle with a given yaw rotation.
        self.gates = []
        for idx, theta in enumerate(angles):
            if idx % 2 == 0:
                z = np.random.uniform(3.5, 4.5) # high altitude range
            else:
                z = np.random.uniform(1.5, 2.5) # low altitude range
    
            center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
            gate = {"center": center, "yaw": theta}
            self.gates.append(gate)

        # Define targets as the centers of the gates.
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])
        self.num_targets = self.gate_positions.shape[0]
        
        # Current target index. Starts at 0.
        self.current_target_index = 0

        # Threshold to consider the target reached.
        self.target_threshold = 0.5
        self.vel_threshold = 0.2  # target reached only if low velocity too.

        # Inactivity penalty parameters.
        self.min_active_vel = 0.2       # Minimum velocity that is considered active.
        self.k_inactivity = 0.5         # Coefficient for inactivity penalty.
        
        # Observation space: We'll use 3 numbers for relative position,
        # 1 number for norm distance, 3 for velocity, 1 for projected velocity,
        # plus additional 5 values for one-hot target index.
        self.obs_dim = 3 + 1 + 3 + 1 + self.num_targets
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action space: acceleration in 3 dimensions.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Maximum values for normalization
        self.max_distance = 10.0
        self.max_vel = 10.0
        self.out_of_bounds = 15.0
        
        # Reward shaping coefficients
        self.k_delta = 0.5
        self.k_vel = 0.1
        self.k_dev = 0.2
        
        self.prev_distance = None

    def reset(self, *, seed=None, return_info=False, options=None):
        # Reset drone position and velocity.
        # self.position = np.zeros(3, dtype=np.float32)
        # self.velocity = np.zeros(3, dtype=np.float32)
        # self.current_step = 0
        # self.current_target_index = 0  # start from the first target.
        # # Compute initial distance to the first target.
        # self.prev_distance = np.linalg.norm(self.gate_positions[self.current_target_index] - self.position)
        

        # Randomly initialize the drone's position within [-max_distance, max_distance] in each axis.
        self.position = np.random.uniform(low=-self.max_distance, high=self.max_distance, size=3).astype(np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.current_step = 0
        
        # Compute distances from the initialized position to all targets.
        distances = np.linalg.norm(self.gate_positions - self.position, axis=1)
        # Set current_target_index to the nearest target.
        self.current_target_index = int(np.argmin(distances))
        self.prev_distance = distances[self.current_target_index]
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = 0.5 * action  # Limit the maximum acceleration to 0.5

        self.current_step += 1
        done, truncated = False, False
        
        # Dynamics update:
        acceleration = action
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

        # Get current target.
        current_gate_position = self.gate_positions[self.current_target_index]
        current_gate_yaw = self.gate_yaws[self.current_target_index]

        # Compute relative position and its norm.
        rel_pos = current_gate_position - self.position
        distance = np.linalg.norm(rel_pos)

        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)

        # Base reward: closer distance yields a higher reward.
        base_reward = 1.0 - 2.0 * (distance / self.max_distance)
        base_reward = np.clip(base_reward, -1.0, 1.0)

        # Distance improvement bonus.
        delta_distance = self.prev_distance - distance
        bonus_distance = 1.0 * delta_distance  # You may scale this factor as needed.

        # Velocity (activity) check.
        vel_norm = np.linalg.norm(self.velocity)
        # Velocity penalty is commented out; you could alternatively add:
        # vel_penalty = self.k_vel * (vel_norm / self.max_vel)

        inactivity_penalty = 0.0
        if vel_norm < self.min_active_vel:
            inactivity_penalty = self.k_inactivity * (self.min_active_vel - vel_norm)

        # Deviation penalty penalizes movement that is not aligned with the target.
        if vel_norm > 1e-6:
            cos_angle = np.dot(self.velocity, unit_rel_pos) / (vel_norm + 1e-6)
            deviation = 1.0 - cos_angle
        else:
            deviation = 0.0
        deviation_penalty = self.k_dev * deviation

        # Final reward structure.
        reward = bonus_distance - deviation_penalty - inactivity_penalty

        # Calculate relative position in the horizontal (xy) plane.
        dx = self.position[0] - current_gate_position[0]
        dy = self.position[1] - current_gate_position[1]

        # Rotate the relative coordinates by -gate_yaw to align the gate with the axes.
        cos_yaw = np.cos(-current_gate_yaw)
        sin_yaw = np.sin(-current_gate_yaw)
        rx = cos_yaw * dx - sin_yaw * dy
        ry = sin_yaw * dx + cos_yaw * dy
        
        in_xy = (abs(rx) <= self.passing_tolerance) and (abs(ry) <= self.passing_tolerance)
        
        # Check altitude (z-axis) separately.
        z_diff = abs(self.position[2] - current_gate_position[2])
        in_z = (z_diff <= self.passing_tolerance)
        
        # The target is reached if both the horizontal and vertical checks pass.
        target_reached = in_xy and in_z

        # Check if current target reached:
        if target_reached:
            # Give bonus for reaching target.
            reward = max(reward, 1.0)
            # Move to the next target in cyclic order.
            self.current_target_index = (self.current_target_index + 1) % self.num_targets
            # Optionally, update any additional state reflecting change of target.
            target_reached = False

        reward = np.clip(reward, -1.0, 1.0)
        # Terminate episode if drone is too far from all targets.
        all_distances = np.linalg.norm(self.gate_positions - self.position, axis=1)
        if np.min(all_distances) > self.out_of_bounds:
            done = True
        
        self.prev_distance = distance

        if self.current_step >= self.max_steps:
            done = True
            truncated = True
        
        
        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        """
        Assemble observation:
          - Relative position (3 values) normalized by max_distance.
          - Norm distance (1 value) normalized.
          - Velocity (3 values) normalized by max_vel.
          - Projected velocity along target direction (1 value)
          - One-hot encoding for current target of length self.num_targets.
        """
        current_target = self.gate_positions[self.current_target_index]
        rel_pos = current_target - self.position
        norm_rel_pos = rel_pos / self.max_distance
        
        distance = np.linalg.norm(rel_pos)
        norm_distance = np.array([distance / self.max_distance], dtype=np.float32)
        norm_velocity = self.velocity / self.max_vel
        
        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)
        
        projected_vel = np.array([np.dot(self.velocity, unit_rel_pos) / self.max_vel], dtype=np.float32)
        
        # One-hot encoding for the current target.
        one_hot = np.zeros(self.num_targets, dtype=np.float32)
        one_hot[self.current_target_index] = 1.0
        
        obs = np.concatenate([norm_rel_pos, norm_distance, norm_velocity, projected_vel, one_hot])
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
    
    def render(self, mode='human'):
        current_target = self.gate_positions[self.current_target_index]
        distance = np.linalg.norm(current_target - self.position)
        print(f"Step: {self.current_step} | Position: {self.position} | Velocity: {self.velocity} | "
              f"Current Index: {self.current_target_index} Target: {current_target} | Dist: {distance:.2f}")
    
    def close(self):
        pass

