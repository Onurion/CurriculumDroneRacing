import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker

class DroneVelGateEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_gates:int=5):
        super(DroneVelGateEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 6000
        self.current_step = 0
        self.gate_size = 1.0  # 1m x 1m gate
        self.passing_tolerance = self.gate_size / 2.0

        self.drone = DroneDynamics(dt=self.dt)
        # PD gains (tuned values) are passed to the low-level controller.
        self.low_level_controller = LowLevelController(self.drone,
                                             Kp_pos=np.array([5.0, 5.0, 5.0]),
                                             Kd_pos=np.array([0.05, 0.05, 0.05]),
                                             Kp_att=np.array([0.1, 0.1, 0.1]),
                                             Kd_att=np.array([0.05, 0.05, 0.05]))

        K_v=np.array([0.42, 0.42, 0.42])
        K_vd=np.array([2.0, 2.0, 2.0])


        self.vel_tracker = VelocityTracker(
            self.drone,
            self.low_level_controller,
            K_v=K_v,
            K_vd=K_vd
        )

        # Define 5 target positions arranged in a circle (for simplicity, in the XY plane with different altitudes)
        angles = np.linspace(0, 2*np.pi, n_gates + 1)[:-1]  # 5 evenly spaced angles

        radius = 5.0
        self.n_gates = n_gates

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
        self.obs_dim = 13 + self.num_targets
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
        position = np.random.uniform(low=-self.max_distance, high=self.max_distance, size=3).astype(np.float32)
        velocity = np.zeros(3, dtype=np.float32)
        self.current_step = 0
        self.drone.reset()
        self.vel_tracker.prev_error = np.zeros(3, dtype=np.float32)
        self.drone.state[0:3] = np.copy(position)

        self.position = position
        self.velocity = velocity

        # Compute distances from the initialized position to all targets.
        distances = np.linalg.norm(self.gate_positions - position, axis=1)
        # Set current_target_index to the nearest target.
        self.current_target_index = int(np.argmin(distances))
        self.prev_distance = distances[self.current_target_index]
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = 0.5*action

        state = self.drone.state
        velocity = state[3:6]

        desired_velocity = velocity + action * self.dt

        # Use the low-level controller to convert the high-level action into control commands.
        control = self.vel_tracker.compute_control(np.array(desired_velocity))
        drone_state, drone_state_dot = self.drone.step(control)

        self.current_step += 1
        done, truncated = False, False

        position = drone_state[:3]
        velocity = drone_state[3:6]
        roll, pitch, yaw = drone_state[6], drone_state[7], drone_state[8]

        self.position = position
        self.velocity = velocity

        # Get current target.
        current_gate_position = self.gate_positions[self.current_target_index]
        current_gate_yaw = self.gate_yaws[self.current_target_index]

        # Compute relative position and its norm.
        rel_pos = current_gate_position - position
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
        bonus_distance = 2.0 * delta_distance  # You may scale this factor as needed.

        # Velocity (activity) check.
        vel_norm = np.linalg.norm(velocity)
        # Velocity penalty is commented out; you could alternatively add:
        # vel_penalty = self.k_vel * (vel_norm / self.max_vel)

        inactivity_penalty = 0.0
        if vel_norm < self.min_active_vel:
            inactivity_penalty = self.k_inactivity * (self.min_active_vel - vel_norm)

        # Deviation penalty penalizes movement that is not aligned with the target.
        if vel_norm > 1e-6:
            cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
            deviation = 1.0 - cos_angle
        else:
            deviation = 0.0
        deviation_penalty = self.k_dev * deviation

        # Final reward structure.
        reward = bonus_distance - deviation_penalty - inactivity_penalty

        # Calculate relative position in the horizontal (xy) plane.
        dx = position[0] - current_gate_position[0]
        dy = position[1] - current_gate_position[1]

        # Rotate the relative coordinates by -gate_yaw to align the gate with the axes.
        cos_yaw = np.cos(-current_gate_yaw)
        sin_yaw = np.sin(-current_gate_yaw)
        rx = cos_yaw * dx - sin_yaw * dy
        ry = sin_yaw * dx + cos_yaw * dy

        in_xy = (abs(rx) <= self.passing_tolerance) and (abs(ry) <= self.passing_tolerance)

        # Check altitude (z-axis) separately.
        z_diff = abs(position[2] - current_gate_position[2])
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
            if self.n_gates == 1:
                done = True

        reward = np.clip(reward, -1.0, 1.0)
        # Terminate episode if drone is too far from all targets.
        all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
        if np.min(all_distances) > self.out_of_bounds:
            done = True

        self.prev_distance = distance

        if abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
            done = True
            reward = -1.0

        if self.current_step >= self.max_steps:
            done = True
            truncated = True


        return self._get_obs(), reward, done, truncated, {}


    def _get_obs(self):
        """
        Assemble observation vector that combines drone's state with gate/target information.
        The observation includes:
        - Relative position (3 values) normalized by max_distance.
        - Norm distance (1 value) normalized.
        - Drone velocity (3 values) normalized by max_vel.
        - Projected velocity along the target direction (1 value).
        - One-hot encoding for current target (num_targets values).
        - Drone absolute position (3 values) normalized by max_distance.
        - Orientation error as sin and cos of yaw difference (2 values).
        Total length = 3 + 1 + 3 + 1 + num_targets + 3 + 2.
        """
        # Get current target position and yaw.
        state = self.drone.state
        current_target = self.gate_positions[self.current_target_index]
        current_gate_yaw = self.gate_yaws[self.current_target_index]
        position = state[0:3]
        velocity = state[3:6]
        yaw = state[8]

        # Compute relative position and normalized distance.
        rel_pos = current_target - position
        norm_rel_pos = rel_pos / self.max_distance
        distance = np.linalg.norm(rel_pos)
        norm_distance = np.array([distance / self.max_distance], dtype=np.float32)

        # Compute normalized drone velocity.
        norm_velocity = velocity / self.max_vel

        # Compute projected velocity along the direction to the target.
        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)
        projected_vel = np.array([np.dot(velocity, unit_rel_pos) / self.max_vel], dtype=np.float32)

        # One-hot encoding for current target gate.
        one_hot = np.zeros(self.num_targets, dtype=np.float32)
        one_hot[self.current_target_index] = 1.0

        # Include drone’s own absolute position normalized by max_distance.
        norm_position = position / self.max_distance

        # Include a simple orientation metric.
        # For example, assume self.yaw represents the drone's current yaw.
        # Compute the difference between drone yaw and current gate yaw.
        yaw_diff = yaw - current_gate_yaw
        sin_yaw = np.array([np.sin(yaw_diff)], dtype=np.float32)
        cos_yaw = np.array([np.cos(yaw_diff)], dtype=np.float32)

        # Assemble all parts into one observation vector.
        obs = np.concatenate([
            norm_rel_pos,      # 3 values
            norm_distance,     # 1 value
            norm_velocity,     # 3 values
            projected_vel,     # 1 value
            one_hot,           # num_targets values
            norm_position,     # 3 values
            sin_yaw,           # 1 value
            cos_yaw            # 1 value
        ])

        # Clip observation to the predefined observation space limits.
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def render(self, mode='human'):
        current_target = self.gate_positions[self.current_target_index]
        distance = np.linalg.norm(current_target - self.position)
        print(f"Step: {self.current_step} | Position: {self.position} | "
              f"Current Index: {self.current_target_index} Target: {current_target} | Dist: {distance:.2f}")

    def close(self):
        pass

