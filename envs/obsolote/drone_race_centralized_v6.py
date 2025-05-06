import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym import RewardWrapper
from stable_baselines3.common.callbacks import BaseCallback
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker


class DroneRaceCentralizedEnv(gym.Env):
    """
    Centralized drone racing environment.
    Two drones race against each other.
    The environment returns a single joint observation (for both drones) and expects
    one joint action (6 values: 3 for each drone). The rewards are computed per drone,
    and here we return a dictionary of rewards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_gates: int = 5):
        super(DroneRaceCentralizedEnv, self).__init__()

        # Simulation parameters.
        self.dt = 0.1
        self.max_steps = 6000
        self.current_step = 0
        self.gate_size = 1.0  # 1m x 1m gate
        self.passing_tolerance = self.gate_size / 2.0

        # Create two drones for the race.
        self.agents = ["drone0", "drone1"]

        # Create a dictionary to hold each drone's components and info.
        self.vehicles = {}
        for drone_id in self.agents:
            drone = DroneDynamics(dt=self.dt)
            low_level_controller = LowLevelController(
                drone,
                Kp_pos=np.array([5.0, 5.0, 5.0]),
                Kd_pos=np.array([0.05, 0.05, 0.05]),
                Kp_att=np.array([0.1, 0.1, 0.1]),
                Kd_att=np.array([0.05, 0.05, 0.05])
            )
            K_v = np.array([0.42, 0.42, 0.42])
            K_vd = np.array([2.0, 2.0, 2.0])
            vel_tracker = VelocityTracker(drone, low_level_controller, K_v=K_v, K_vd=K_vd)
            self.vehicles[drone_id] = {
                "drone": drone,
                "low_level_controller": low_level_controller,
                "vel_tracker": vel_tracker,
                "current_target_index": 0,   # Which gate is the target
                "prev_distance": None,       # Last recorded distance to current target
                "progress": 0,               # Number of gates passed
                "prev_progress": 0           # For checking overtaking progress
            }

        # Define the gates. Here, gates are arranged in a circle.
        angles = np.linspace(0, 2 * np.pi, n_gates + 1)[:-1]
        radius = 5.0
        self.n_gates = n_gates
        self.gates = []
        for idx, theta in enumerate(angles):
            if idx % 2 == 0:
                z = np.random.uniform(3.5, 4.5)  # high altitude range
            else:
                z = np.random.uniform(1.5, 2.5)  # low altitude range
            center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
            gate = {"center": center, "yaw": theta}
            self.gates.append(gate)
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])
        self.num_targets = self.gate_positions.shape[0]

        # Thresholds and reward shaping parameters.
        self.target_threshold = 0.5
        self.min_active_vel = 1.0

        self.max_distance = 10.0
        self.max_vel = 10.0

        # Observation dimensions per drone.
        # From the original _get_obs_agent: 3 + 1 + 3 + 1 + num_targets + 3 + 2 + 3 + 1 = 17 + num_targets.
        self.obs_dim = 17 + self.num_targets
        self.central_obs_dim = 2 * self.obs_dim

        # Define centralized observation and action spaces.
        # Joint observation: concatenation of both drone observations.
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.central_obs_dim,), dtype=np.float32)
        # Joint action: 6 values (3 for each).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Collision threshold between drones.
        self.collision_distance = 0.5  # in meters

        # Tuning parameters for reward shaping.
        self.a = 1.0                   # Controls the shape of the exponential decay.
        self.w_base = 0.25             # Weight for base_reward.
        self.w_bonus = 0.5             # Weight for bonus_distance.
        self.w_dev = 0.3               # Weight for deviation penalty.
        self.w_inactivity = 0.4        # Weight for inactivity penalty.

    def reset(self, *, seed=None, return_info=False, options=None):
        self.current_step = 0
        # Reset each drone’s state.
        for drone_id in self.agents:
            position = np.random.uniform(low=-self.max_distance, high=self.max_distance, size=3).astype(np.float32)
            # Start with zero velocity.
            vehicle = self.vehicles[drone_id]
            vehicle["drone"].reset()
            vehicle["vel_tracker"].prev_error = np.zeros(3, dtype=np.float32)
            vehicle["drone"].state[0:3] = np.copy(position)

            # Initialize target based on the nearest gate.
            distances = np.linalg.norm(self.gate_positions - position, axis=1)
            vehicle["current_target_index"] = int(np.argmin(distances))
            vehicle["prev_distance"] = distances[vehicle["current_target_index"]]
            vehicle["progress"] = 0
            vehicle["prev_progress"] = 0

        # Return a centralized observation by concatenating individual observations.
        central_obs = np.concatenate([self._get_obs_drone(drone_id) for drone_id in self.agents])
        return central_obs, {}

    def step(self, actions):
        """
        Expects a single joint action of shape (6,): first 3 values for drone0, next 3 for drone1.
        Returns:
          - joint_obs: concatenated observation for both drones.
          - rewards: a dict with individual rewards for each drone.
          - done: a boolean indicating global termination.
          - info: additional information (e.g., progress).
        """
        rewards = {}
        dones = {}
        # We set a flag if any individual condition (out-of-bound, crash) causes termination.
        global_done = False
        truncated = False
        infos = {}

        # Split joint action.
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        action_drone0 = 1.0 * actions[:3]
        action_drone1 = 1.0 * actions[3:]
        joint_actions = {"drone0": action_drone0, "drone1": action_drone1}

        # Process each drone.
        for drone_id in self.agents:
            act = joint_actions[drone_id]
            vehicle = self.vehicles[drone_id]
            drone = vehicle["drone"]
            state = drone.state
            velocity = state[3:6]
            desired_velocity = velocity + act * self.dt
            control = vehicle["vel_tracker"].compute_control(np.array(desired_velocity))
            new_state, _ = drone.step(control)
            position = new_state[:3]
            velocity = new_state[3:6]
            roll, pitch, yaw = new_state[6], new_state[7], new_state[8]

            # Determine current target.
            current_target_idx = vehicle["current_target_index"]
            current_gate_position = self.gate_positions[current_target_idx]
            current_gate_yaw = self.gate_yaws[current_target_idx]

            # Relative position from drone to target.
            rel_pos = current_gate_position - position
            distance = np.linalg.norm(rel_pos)
            if distance > 1e-6:
                unit_rel_pos = rel_pos / distance
            else:
                unit_rel_pos = np.zeros_like(rel_pos)

            # Base reward: normalized exponential shaping.
            x = distance / self.max_distance
            base_reward = 2 * (np.exp(-self.a * x) - np.exp(-self.a)) / (1 - np.exp(-self.a)) - 1
            delta_distance = vehicle["prev_distance"] - distance
            bonus_distance = 2.5 * delta_distance
            vel_norm = np.linalg.norm(velocity)

            inactivity_penalty = 0.0
            if vel_norm < self.min_active_vel:
                inactivity_penalty = (self.min_active_vel - vel_norm)
            if vel_norm > 1e-6:
                cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
                deviation = 1.0 - cos_angle
            else:
                deviation = 0.0

            # Combine reward components.
            reward = (self.w_base * base_reward +
                      self.w_bonus * bonus_distance -
                      self.w_dev * deviation -
                      self.w_inactivity * inactivity_penalty)

            # Check if the target gate is reached.
            dx = position[0] - current_gate_position[0]
            dy = position[1] - current_gate_position[1]
            cos_yaw = np.cos(-current_gate_yaw)
            sin_yaw = np.sin(-current_gate_yaw)
            rx = cos_yaw * dx - sin_yaw * dy
            ry = sin_yaw * dx + cos_yaw * dy
            in_xy = (abs(rx) <= self.passing_tolerance) and (abs(ry) <= self.passing_tolerance)
            z_diff = abs(position[2] - current_gate_position[2])
            in_z = (z_diff <= self.passing_tolerance)
            target_reached = in_xy and in_z

            if target_reached:
                reward = 1.0
                vehicle["current_target_index"] = (vehicle["current_target_index"] + 1) % self.num_targets
                vehicle["progress"] += 1


            drone_done = False
            # Check out-of-bound condition.
            all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
            if np.min(all_distances) > self.max_distance:
                drone_done = True
                reward = -1.0

            # Check for excessive roll or pitch.
            if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
                drone_done = True
                reward = -1.0

            # Save the current distance for shaping.
            vehicle["prev_distance"] = distance
            dones[drone_id] = drone_done
            rewards[drone_id] = np.clip(reward, -1.0, 1.0)
            infos[drone_id] = {"progress": vehicle["progress"], "position": position}

        # --- Overtaking Bonus ---
        progress0 = self.vehicles["drone0"]["progress"]
        progress1 = self.vehicles["drone1"]["progress"]
        if progress0 > progress1:
            rewards["drone0"] = np.clip(rewards["drone0"] + 0.2, -1.0, 1.0)
        elif progress1 > progress0:
            rewards["drone1"] = np.clip(rewards["drone1"] + 0.2, -1.0, 1.0)

        # (Optional) --- Collision Penalty ---
        # pos0 = self.vehicles["drone0"]["drone"].state[:3]
        # pos1 = self.vehicles["drone1"]["drone"].state[:3]
        # if np.linalg.norm(pos0 - pos1) < self.collision_distance:
        #     for drone_id in self.agents:
        #         rewards[drone_id] = np.clip(rewards[drone_id] - 0.5, -1.0, 1.0)

        
        # For reward, you might take a weighted sum or average.
        # Here, for example, we average the two rewards.
        combined_reward = (rewards["drone0"] + rewards["drone1"]) / 2.0

        # For termination, you may decide to end the entire episode
        # only if both drones have failed.
        global_done = dones["drone0"] and dones["drone1"]

        self.current_step += 1
        if self.current_step >= self.max_steps:
            global_done = True
            truncated = True

        # Construct joint (centralized) observation.
        central_obs = np.concatenate([self._get_obs_drone(drone_id) for drone_id in self.agents])
        return central_obs, combined_reward, global_done, truncated,  infos

    def _get_obs_drone(self, drone_id):
        """
        Assemble an observation vector for a given drone.
        The observation includes:
          - Relative position (3 values) normalized by max_distance.
          - Norm distance (1 value) normalized.
          - Drone velocity (3 values) normalized by max_vel.
          - Projected velocity along the target direction (1 value).
          - One-hot encoding for current target (num_targets values).
          - Drone absolute position (3 values) normalized by max_distance.
          - Orientation error as sin and cos of yaw difference (2 values).
          - Relative opponent information (3 values for normalized opponent relative position
            and 1 value for normalized opponent distance).
        Total length = 17 + num_targets.
        """
        vehicle = self.vehicles[drone_id]
        drone = vehicle["drone"]
        state = drone.state
        current_target_idx = vehicle["current_target_index"]
        current_target = self.gate_positions[current_target_idx]
        current_gate_yaw = self.gate_yaws[current_target_idx]
        position = state[0:3]
        velocity = state[3:6]
        yaw = state[8]

        # Relative position to current target.
        rel_pos = current_target - position
        norm_rel_pos = rel_pos / self.max_distance
        distance = np.linalg.norm(rel_pos)
        norm_distance = np.array([distance / self.max_distance], dtype=np.float32)
        norm_velocity = velocity / self.max_vel

        if distance > 1e-6:
            unit_rel_pos = rel_pos / distance
        else:
            unit_rel_pos = np.zeros_like(rel_pos)
        projected_vel = np.array([np.dot(velocity, unit_rel_pos) / self.max_vel], dtype=np.float32)

        one_hot = np.zeros(self.num_targets, dtype=np.float32)
        one_hot[current_target_idx] = 1.0

        norm_position = position / self.max_distance
        yaw_diff = yaw - current_gate_yaw
        sin_yaw = np.array([np.sin(yaw_diff)], dtype=np.float32)
        cos_yaw = np.array([np.cos(yaw_diff)], dtype=np.float32)

        # Opponent information.
        opponent_id = "drone1" if drone_id == "drone0" else "drone0"
        opponent_position = self.vehicles[opponent_id]["drone"].state[0:3]
        opponent_rel_pos = opponent_position - position
        normalized_opponent_rel_pos = opponent_rel_pos / self.max_distance
        opponent_distance = np.linalg.norm(opponent_rel_pos)
        normalized_opponent_distance = np.array([opponent_distance / self.max_distance], dtype=np.float32)

        obs = np.concatenate([
            norm_rel_pos,                   # 3 values
            norm_distance,                  # 1 value
            norm_velocity,                  # 3 values
            projected_vel,                  # 1 value
            one_hot,                        # num_targets values
            norm_position,                  # 3 values
            sin_yaw,                        # 1 value
            cos_yaw,                        # 1 value
            normalized_opponent_rel_pos,    # 3 values
            normalized_opponent_distance    # 1 value
        ])

        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        return obs

    def render(self, mode='human'):
        for agent in self.agents:
            pos = self.vehicles[agent]["drone"].state[:3]
            current_target_idx = self.vehicles[agent]["current_target_index"]
            current_target = self.gate_positions[current_target_idx]
            distance = np.linalg.norm(current_target - pos)
            print(f"Agent: {agent} | Dist: {distance:.2f} |  Target Index: {self.vehicles[agent]['current_target_index']} | " 
                  f"Target: {current_target} | Step: {self.current_step} | Position: {pos} ")
            

    def get_state(self, agent):
        return self.vehicles[agent]["drone"].state