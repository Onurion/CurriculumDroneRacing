import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym import RewardWrapper
from stable_baselines3.common.callbacks import BaseCallback
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker


class DroneRaceCentralizedMultiEnv(gym.Env):
    """
    Centralized drone racing environment.
    Two drones race against each other.
    The environment returns a single joint observation (for both drones) and expects
    one joint action (6 values: 3 for each drone). The rewards are computed per drone,
    and here we return a dictionary of rewards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_agents: int=2,
                 n_gates: int = 5,
                 radius: float = 10.0,
                 minimum_velocity: float=0.2,
                 action_coefficient: float=1.0,
                 distance_exp_decay: float=2.0,
                 w_distance: float=0.0,
                 w_distance_change: float=1.0,
                 w_alignment: float=0.5,
                 w_deviation: float=0.0,
                 w_inactivity: float=0.0,
                 w_collision_penalty: float=0.0,
                 reward_type: int=2,
                 observation_type: int=2,
                 buffer_size: int=10,
                 is_buffer_obs: bool=False,
                 random_init: bool=True,
                 max_steps: int = 2000,
                 dt: float = 0.1,
                 gate_size: float = 1.0,
                 # Curriculum parameters:
                 enable_collision: bool=False,
                 terminate_on_collision: bool=False,
                 collision_penalty: float = 0.5,
                 drone_collision_margin: float =0.5,
                 gate_passing_tolerance: float = 0.5,
                 enable_takeover:bool=False,
                 takeover_reward: float =0.0,
                 jitter_range:list = [0.25, 0.75]):
        super(DroneRaceCentralizedMultiEnv, self).__init__()


        # Simulation parameters.
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        self.gate_size = gate_size  # 1m x 1m gate
        self.gate_passing_tolerance = gate_passing_tolerance
        self.random_init = random_init
        self.jitter_range = jitter_range

        # Curriculum parameters.
        self.action_coefficient = action_coefficient          # Multiplier applied to the raw action.
        self.minimum_velocity = minimum_velocity     # Minimum velocity (used to penalize inactivity).
        self.enable_collision = enable_collision  # Whether collision detection is enabled.
        self.collision_penalty = collision_penalty  # Penalty for collision.
        self.drone_collision_margin = drone_collision_margin  # Drone Collision threshold in meters.
        self.terminate_on_collision = terminate_on_collision  # Terminate episode on collision.
        self.enable_takeover = enable_takeover    # Whether takeover bonus is enabled.
        self.takeover_reward = takeover_reward    # Reward bonus for overtaking in selfplay.
        

        # Reward parameters.
        self.a = distance_exp_decay                   # Controls the shape of the exponential decay.
        self.w_distance = w_distance             # Weight for base_reward.
        self.w_distance_change = w_distance_change             # Weight for bonus_distance.
        self.w_deviation = w_deviation            # Weight for deviation penalty.
        self.w_inactivity = w_inactivity       # Weight for inactivity penalty.
        self.w_alignment = w_alignment
        self.w_collision_penalty = w_collision_penalty  # Weight for collision penalty.

        self.is_buffer_obs = is_buffer_obs

        self.agents = []
        self.n_agents = n_agents
        # Define two drones (agents) for self-play.
        for i in range(self.n_agents):
            self.agents.append(f"drone{i}")

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

        self.n_gates = int(n_gates)
        # Define the gates. Here, gates are arranged in a circle.
        angles = np.linspace(0, 2 * np.pi, self.n_gates + 1)[:-1]

        self.gates = []
        for idx, theta in enumerate(angles):
            if idx % 2 == 0:
                z = 4 # np.random.uniform(3.5, 4.5)  # high altitude range
            else:
                z = 3 # np.random.uniform(1.5, 2.5)  # low altitude range
            center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
            gate = {"center": center, "yaw": theta}
            self.gates.append(gate)
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])


        # Thresholds and reward shaping parameters.
        self.target_threshold = 0.5

        self.max_distance = 10.0
        self.max_vel = 10.0

        # Observation: the observation vector length is 17 + num_targets.
        self.num_targets = self.gate_positions.shape[0]
        self.reward_type = reward_type

        # Reward function selector.
        self.reward_function = self.reward_function_1 if reward_type == 1 else self.reward_function_2
        if observation_type == 1:
            self.obs_dim = 13 + 4 * (self.n_agents - 1) + self.num_targets
            self.observation_function = self._get_obs_1
        else:
            self.obs_dim = 24 + 9 * (self.n_agents - 1) + self.num_targets
            self.observation_function = self._get_obs_2

        self.central_obs_dim = self.n_agents * self.obs_dim

        # Define centralized observation and action spaces.
        # Joint observation: concatenation of both drone observations.
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.central_obs_dim,), dtype=np.float32)
        # Joint action: 6 values (3 for each).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3*self.n_agents,), dtype=np.float32)

    def sample_safe_start_position(self):
        low = np.array([-self.max_distance, -self.max_distance, 0.0])
        high = np.array([self.max_distance,  self.max_distance,  self.max_distance])
        safe_distance = self.gate_size  # or choose a fraction/multiple of self.gate_size

        while True:
            position = np.random.uniform(low=low, high=high).astype(np.float32)
            safe = True

            # Check against all gate positions
            for gate_pos in self.gate_positions:
                # Convert gate position to a numpy array, in case it's not already
                gate_pos = np.array(gate_pos)
                # Consider only the lateral plane (e.g., x and y) for gate boundaries.
                # (If needed, you can include the vertical (z) difference as well.)
                distance_xy = np.linalg.norm(position - gate_pos)
                if distance_xy < safe_distance:
                    safe = False
                    break  # no need to check other gates

            if safe:
                return position

    def initialize_drones_closer(self, current_gate = 1, offset_distance = 2.0, jitter_range = 0.5):
        # Get the first gate (gate_0)
        gate0 = self.gates[current_gate]
        gate0_center = gate0["center"]   # Center of the first gate
        gate0_yaw = gate0["yaw"]         # Yaw of the first gate

        # Define a fixed offset distance "before" the gate.
        # This offset moves the drone along the opposite direction of the gate's facing.
        # Compute the unit direction vector from the gate heading.
        # (We use the yaw angle to determine the direction in the XY plane.)
        direction_vector = np.array(
            [np.cos(gate0_yaw), np.sin(gate0_yaw), 0.0],
            dtype=np.float32
        )

        # Compute the base starting position
        base_position = gate0_center - offset_distance * direction_vector

        # Initialize all drones near this base position with a small random jitter.
        # Define a jitter range (meters) to spread the drones slightly.

        jitter = np.random.uniform(-jitter_range, jitter_range, size=3).astype(np.float32)

        # Optionally, keep the vertical (z) position constant.
        # jitter[2] = 0.0

        # Finalized starting position for this drone
        position = base_position + jitter

        return position

    def reset(self, *, seed=None, return_info=False, options=None):
        self.current_step = 0

        offset_distance = np.random.uniform(low=1.5, high=3.0)
        jitter_range = np.random.uniform(low=self.jitter_range[0], high=self.jitter_range[1])
        gate_index = np.random.randint(low=0, high=len(self.gate_positions))

        for agent in self.agents:
            # Randomly initialize the drone's position.
            # Use this version:
            if self.random_init:
                position = self.sample_safe_start_position()
                # low = np.array([-self.max_distance, -self.max_distance, 0.0])
                # high = np.array([self.max_distance,  self.max_distance,  self.max_distance])
                # position = np.random.uniform(low=low, high=high).astype(np.float32)
            else:
                position = self.initialize_drones_closer(current_gate=gate_index, offset_distance=offset_distance, jitter_range=jitter_range)


            position[2] = max(0, position[2])
            velocity = np.zeros(3, dtype=np.float32)
            vehicle = self.vehicles[agent]
            vehicle["drone"].reset()
            vehicle["vel_tracker"].prev_error = np.zeros(3, dtype=np.float32)
            vehicle["drone"].state[0:3] = np.copy(position)
            vehicle["drone"].state[3:6] = np.copy(velocity)

            # Determine the nearest gate as the first target.
            distances = np.linalg.norm(self.gate_positions - position, axis=1)
            vehicle["current_target_index"] = int(np.argmin(distances))
            vehicle["prev_distance"] = distances[vehicle["current_target_index"]]
            vehicle["prev_velocity"] = np.zeros(3, dtype=np.float32)
            vehicle["progress"] = 0
            vehicle["prev_progress"] = 0

        # Return a centralized observation by concatenating individual observations.
        central_obs = np.concatenate([self.observation_function(drone_id) for drone_id in self.agents])
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
        # action_drone0 = 1.0 * actions[:3]
        # action_drone1 = 1.0 * actions[3:]
        # joint_actions = {"drone0": action_drone0, "drone1": action_drone1}

        actions_per_drone = len(actions) // self.n_agents  # Assuming equal action split

        joint_actions = {}

        for i, drone_id in enumerate(self.vehicles.keys()):
            start_idx = i * actions_per_drone
            end_idx = start_idx + actions_per_drone
            joint_actions[drone_id] = 1.0 * actions[start_idx:end_idx]  # Scale if needed

        # print ("Joint Actions: ", joint_actions)

        # Process each drone.
        for drone_id in self.agents:
            act = joint_actions[drone_id]
            vehicle = self.vehicles[drone_id]
            drone = vehicle["drone"]
            state = drone.state
            velocity = state[3:6]
            # print ("Velocity: ", velocity, " action: ", act)
            vel_norm = np.linalg.norm(velocity)
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

             # Compute reward based on the chosen reward function.
            reward = self.reward_function(distance, vehicle["prev_distance"], velocity, unit_rel_pos)

            # Check if the target gate is reached.
            dx = position[0] - current_gate_position[0]
            dy = position[1] - current_gate_position[1]
            cos_yaw = np.cos(-current_gate_yaw)
            sin_yaw = np.sin(-current_gate_yaw)
            rx = cos_yaw * dx - sin_yaw * dy
            ry = sin_yaw * dx + cos_yaw * dy
            in_xy = (abs(rx) <= self.gate_passing_tolerance) and (abs(ry) <= self.gate_passing_tolerance)
            z_diff = abs(position[2] - current_gate_position[2])
            in_z = (z_diff <= self.gate_passing_tolerance)
            target_reached = in_xy and in_z

            if target_reached:
                reward = 1.0
                vehicle["current_target_index"] = (vehicle["current_target_index"] + 1) % self.num_targets
                vehicle["progress"] += 1
                target_reached = False


            drone_done = False
            # Check out-of-bound condition.
            all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
            if np.min(all_distances) > self.max_distance:
                drone_done = True
                reward = -1.0

            infos[drone_id] = {"progress": vehicle["progress"], "position": position, "velocity": vel_norm, "collision": 0}
            # Check for excessive roll or pitch.
            if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
                drone_done = True
                reward = -1.0
                infos[drone_id]["collision"] = 1

            # Save the current distance for shaping.
            vehicle["prev_distance"] = distance
            dones[drone_id] = drone_done
            rewards[drone_id] = np.clip(reward, -1.0, 1.0)


        # --- Overtaking Bonus ---
        for drone_i in self.vehicles:
            for drone_j in self.vehicles:
                if drone_i != drone_j:
                    if self.vehicles[drone_i]["progress"] > self.vehicles[drone_j]["progress"]:
                        rewards[drone_i] = np.clip(rewards[drone_i] + self.takeover_reward, -1.0, 1.0)

        # (Optional) --- Collision Penalty ---
        drone_ids = list(self.vehicles.keys())
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):  # Avoid redundant checks
                pos_i = self.vehicles[drone_ids[i]]["drone"].state[:3]
                pos_j = self.vehicles[drone_ids[j]]["drone"].state[:3]
                if np.linalg.norm(pos_i - pos_j) < self.drone_collision_margin:
                    # Only mark the specific drones involved in the collision
                    drone_i = drone_ids[i]
                    drone_j = drone_ids[j]
                    
                    # Apply penalty and mark collision for the two colliding drones
                    rewards[drone_i] = np.clip(rewards[drone_i] - self.collision_penalty, -1.0, 1.0)
                    rewards[drone_j] = np.clip(rewards[drone_j] - self.collision_penalty, -1.0, 1.0)
                    
                    infos[drone_i]["collision"] = 1
                    infos[drone_j]["collision"] = 1
                    
                    if self.terminate_on_collision:
                        dones["__all__"] = True

        # Compute the average reward across all drones
        combined_reward = np.mean([rewards[drone_id] for drone_id in self.vehicles])

        # Episode termination logic: all drones must be done for global termination
        global_done = all(dones[drone_id] for drone_id in self.vehicles)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            global_done = True
            truncated = True

        # Construct joint (centralized) observation.
        central_obs = np.concatenate([self.observation_function(drone_id) for drone_id in self.agents])
        return central_obs, combined_reward, global_done, truncated,  infos

    def reward_function_2(self, distance, prev_distance, velocity, unit_rel_pos):
        x = distance / self.max_distance
        base_reward = 2 * (np.exp(-self.a * x) - np.exp(-self.a)) / (1 - np.exp(-self.a)) - 1
        delta_distance = prev_distance - distance
        bonus_distance = 2.5 * delta_distance
        vel_norm = np.linalg.norm(velocity)

        cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
        alignment_component = np.abs(vel_norm - self.minimum_velocity) * cos_angle

        inactivity_penalty = min(vel_norm - self.minimum_velocity, 0)

        reward = (self.w_distance * base_reward +
                  self.w_distance_change * bonus_distance +
                  self.w_alignment * alignment_component +
                  self.w_inactivity * inactivity_penalty)
        return reward

    def reward_function_1(self, distance, prev_distance, velocity, unit_rel_pos):
        x = distance / self.max_distance
        base_reward = 2 * (np.exp(-self.a * x) - np.exp(-self.a)) / (1 - np.exp(-self.a)) - 1
        delta_distance = prev_distance - distance
        bonus_distance = 2.5 * delta_distance
        vel_norm = np.linalg.norm(velocity)

        # if vel_norm < self.minimum_velocity:
        #     inactivity_penalty = (self.minimum_velocity - vel_norm)

        # speed_gap = vel_norm - self.minimum_velocity
        # Define a desired speed threshold (e.g., minimum_velocity + margin)
        desired_speed = self.minimum_velocity
        if vel_norm <= desired_speed:
            # Small bonus if the drone is near the desired speed
            speed_gap = vel_norm - self.minimum_velocity
        else:
            # Penalize speeds exceeding the desired speed
            speed_gap = desired_speed - vel_norm

        if vel_norm > 1e-6:
            cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
            deviation = 1.0 - cos_angle
        else:
            deviation = 0.0

        reward = (self.w_distance * base_reward +
                  self.w_distance_change * bonus_distance -
                  self.w_deviation * deviation +
                  self.w_inactivity * speed_gap)
        return reward
    
    # def _get_obs_drone(self, drone_id):
    #     """
    #     Assemble an observation vector for a given drone.
    #     The observation includes:
    #       - Relative position (3 values) normalized by max_distance.
    #       - Norm distance (1 value) normalized.
    #       - Drone velocity (3 values) normalized by max_vel.
    #       - Projected velocity along the target direction (1 value).
    #       - One-hot encoding for current target (num_targets values).
    #       - Drone absolute position (3 values) normalized by max_distance.
    #       - Orientation error as sin and cos of yaw difference (2 values).
    #       - Relative opponent information (3 values for normalized opponent relative position
    #         and 1 value for normalized opponent distance).
    #     Total length = 17 + num_targets.
    #     """
    #     vehicle = self.vehicles[drone_id]
    #     drone = vehicle["drone"]
    #     state = drone.state
    #     current_target_idx = vehicle["current_target_index"]
    #     current_target = self.gate_positions[current_target_idx]
    #     current_gate_yaw = self.gate_yaws[current_target_idx]
    #     position = state[0:3]
    #     velocity = state[3:6]
    #     yaw = state[8]

    #     # Relative position to current target.
    #     rel_pos = current_target - position
    #     norm_rel_pos = rel_pos / self.max_distance
    #     distance = np.linalg.norm(rel_pos)
    #     norm_distance = np.array([distance / self.max_distance], dtype=np.float32)
    #     norm_velocity = velocity / self.max_vel

    #     if distance > 1e-6:
    #         unit_rel_pos = rel_pos / distance
    #     else:
    #         unit_rel_pos = np.zeros_like(rel_pos)
    #     projected_vel = np.array([np.dot(velocity, unit_rel_pos) / self.max_vel], dtype=np.float32)

    #     one_hot = np.zeros(self.num_targets, dtype=np.float32)
    #     one_hot[current_target_idx] = 1.0

    #     norm_position = position / self.max_distance
    #     yaw_diff = yaw - current_gate_yaw
    #     sin_yaw = np.array([np.sin(yaw_diff)], dtype=np.float32)
    #     cos_yaw = np.array([np.cos(yaw_diff)], dtype=np.float32)

    #     # Opponent information.
    #     opponent_id = "drone1" if drone_id == "drone0" else "drone0"
    #     opponent_position = self.vehicles[opponent_id]["drone"].state[0:3]
    #     opponent_rel_pos = opponent_position - position
    #     normalized_opponent_rel_pos = opponent_rel_pos / self.max_distance
    #     opponent_distance = np.linalg.norm(opponent_rel_pos)
    #     normalized_opponent_distance = np.array([opponent_distance / self.max_distance], dtype=np.float32)

    #     obs = np.concatenate([
    #         norm_rel_pos,                   # 3 values
    #         norm_distance,                  # 1 value
    #         norm_velocity,                  # 3 values
    #         projected_vel,                  # 1 value
    #         one_hot,                        # num_targets values
    #         norm_position,                  # 3 values
    #         sin_yaw,                        # 1 value
    #         cos_yaw,                        # 1 value
    #         normalized_opponent_rel_pos,    # 3 values
    #         normalized_opponent_distance    # 1 value
    #     ])

    #     obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
    #     return obs

    def _get_obs_1(self, agent):
        """
        Assemble an observation vector for the given agent.
        The observation includes:
          - Relative position (3 values) normalized by max_distance.
          - Norm distance (1 value) normalized.
          - Drone velocity (3 values) normalized by max_vel.
          - Projected velocity along the target direction (1 value).
          - One-hot encoding for current target (num_targets values).
          - Drone absolute position (3 values) normalized by max_distance.
          - Orientation error as sin and cos of yaw difference (2 values).
        Total length = 3+1+3+1+num_targets+3+2.
        """
        vehicle = self.vehicles[agent]
        drone = vehicle["drone"]
        state = drone.state
        current_target_idx = vehicle["current_target_index"]
        current_target = self.gate_positions[current_target_idx]
        current_gate_yaw = self.gate_yaws[current_target_idx]
        position = state[0:3]
        velocity = state[3:6]
        yaw = state[8]

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

        # Get information about all other drones (opponents)
        opponents_info = {}
        for other_agent in self.vehicles:
            if other_agent != agent:
                opponents_info[other_agent] = {
                    "position": self.vehicles[other_agent]["drone"].state[0:3]
                }

        # Calculate normalized values for all opponents
        all_opponent_rel_pos = []
        all_opponent_distances = []

        for opponent, info in opponents_info.items():
            # Relative position
            opponent_rel_pos = info["position"] - position
            normalized_rel_pos = opponent_rel_pos / self.max_distance
            all_opponent_rel_pos.append(normalized_rel_pos)

            # Distance
            opponent_distance = np.linalg.norm(opponent_rel_pos)
            normalized_distance = opponent_distance / self.max_distance
            all_opponent_distances.append([normalized_distance])


        # Convert lists to numpy arrays
        all_opponent_rel_pos = np.concatenate(all_opponent_rel_pos)  # 3 values per opponent
        all_opponent_distances = np.concatenate(all_opponent_distances)  # 1 value per opponent
        # 13 + 4*(n_opponents) + n_targets
        obs = np.concatenate([
            norm_rel_pos,         # 3 values
            norm_distance,        # 1 value
            norm_velocity,        # 3 values
            projected_vel,        # 1 value
            one_hot,              # num_targets values
            norm_position,        # 3 values
            sin_yaw,              # 1 value
            cos_yaw,              # 1 value
            all_opponent_rel_pos,  # 3 * (n_opponents) values
            all_opponent_distances  # 1 * (n_opponents) values
        ])

        return obs
    
    def _get_obs_2(self, agent):
        """
        Assemble an observation vector for the given agent.
        The observation includes:
          - Relative position (3 values) normalized by max_distance.
          - Norm distance (1 value) normalized.
          - Drone velocity (3 values) normalized by max_vel.
          - Projected velocity along the target direction (1 value).
          - One-hot encoding for current target (num_targets values).
          - Drone absolute position (3 values) normalized by max_distance.
          - Orientation error as sin and cos of yaw difference (2 values).
          - Opponent relative position (3 values) normalized by max_distance.
          - Opponent distance (1 value) normalized.
        Total length = 3+1+3+1+num_targets+3+2+3+1.
        """
        vehicle = self.vehicles[agent]
        drone = vehicle["drone"]
        state = drone.state
        current_target_idx = vehicle["current_target_index"]
        current_target = self.gate_positions[current_target_idx]
        current_gate_yaw = self.gate_yaws[current_target_idx]
        position = state[0:3]
        velocity = state[3:6]
        yaw = state[8]
        speed = np.linalg.norm(velocity) + 1e-6
        normalized_speed = np.array([speed / self.max_vel], dtype=np.float32)


        # 1. Heading error (angle between agent's heading and the vector to the gate)
        ideal_direction = current_target - position
        ideal_direction_norm = ideal_direction / (np.linalg.norm(ideal_direction) + 1e-6)

        normalized_relative_position = ideal_direction / self.max_distance
        distance = np.linalg.norm(ideal_direction)
        normalized_distance = np.array([distance / self.max_distance], dtype=np.float32)
        normalized_velocity = velocity / self.max_vel


        # Assuming `velocity` indicates current heading direction
        current_direction = velocity / speed
        heading_error = np.arccos(np.clip(np.dot(current_direction, ideal_direction_norm), -1.0, 1.0))
        normalized_heading_error = np.array([heading_error / np.pi], dtype=np.float32)

        angular_speed = np.linalg.norm(state[9:12]) + 1e-6
        normalized_angular_speed = np.array([angular_speed / self.max_vel], dtype=np.float32)
        normalized_angular_velocity = state[9:12] / self.max_vel

        if distance > 1e-6:
            unit_rel_pos = ideal_direction / distance
        else:
            unit_rel_pos = np.zeros_like(ideal_direction)
        projected_vel = np.array([np.dot(velocity, unit_rel_pos) / self.max_vel], dtype=np.float32)
        one_hot = np.zeros(self.num_targets, dtype=np.float32)
        one_hot[current_target_idx] = 1.0
        normalized_position = position / self.max_distance
        yaw_diff = yaw - current_gate_yaw
        sin_yaw = np.array([np.sin(yaw_diff)], dtype=np.float32)
        cos_yaw = np.array([np.cos(yaw_diff)], dtype=np.float32)

        normalized_progress = np.array([self.vehicles[agent]["progress"] / 25.0], dtype=np.float32)
        delta_velocity = velocity - self.vehicles[agent]["prev_velocity"]  # or compute as (current_velocity - previous_velocity) if
        normalized_delta_velocity = delta_velocity / self.max_vel
        normalized_time = np.array([self.current_step / self.max_steps], dtype=np.float32)

        # Get information about all other drones (opponents)
        opponents_info = {}
        for other_agent in self.vehicles:
            if other_agent != agent:
                opponents_info[other_agent] = {
                    "position": self.vehicles[other_agent]["drone"].state[0:3],
                    "velocity": self.vehicles[other_agent]["drone"].state[3:6],
                    "progress": self.vehicles[other_agent]["progress"]
                }

        # Calculate normalized values for all opponents
        all_opponent_rel_pos = []
        all_opponent_distances = []
        all_opponent_velocities = []
        all_opponent_progresses = []

        for opponent, info in opponents_info.items():
            # Relative position
            opponent_rel_pos = info["position"] - position
            opponent_normalized_rel_pos = opponent_rel_pos / self.max_distance
            all_opponent_rel_pos.append(opponent_normalized_rel_pos)

            # Distance
            opponent_distance = np.linalg.norm(opponent_rel_pos)
            opponent_normalized_distance = opponent_distance / self.max_distance
            all_opponent_distances.append([opponent_normalized_distance])

            # Velocity
            opponent_normalized_velocity = info["velocity"] / self.max_vel
            all_opponent_velocities.append(opponent_normalized_velocity)

            # Progress
            opponent_normalized_progress = info["progress"] / 25.0
            all_opponent_progresses.append([opponent_normalized_progress])


        # Convert lists to numpy arrays
        all_opponent_rel_pos = np.concatenate(all_opponent_rel_pos)  # 3 values per opponent
        all_opponent_distances = np.concatenate(all_opponent_distances)  # 1 value per opponent
        all_opponent_velocities = np.concatenate(all_opponent_velocities)  # 3 values per opponent
        all_opponent_progresses = np.concatenate(all_opponent_progresses)  # 1 value per opponent

        # Calculate gate ratios for all opponents
        gate_ratios = []
        for opponent, info in opponents_info.items():
            opponent_progress = info["progress"]
            ratio = (self.vehicles[agent]["progress"] + 1e-6) / (opponent_progress + 1e-6)
            gate_ratios.append([ratio])

        # Convert to numpy array
        gate_ratios = np.concatenate(gate_ratios, dtype=np.float32)  # 1 value per opponent



        # Update the observation space concatenation 24 + 9*(n_opponents)
        obs = np.concatenate([
            normalized_relative_position,      # 3 values
            normalized_distance,               # 1 value
            normalized_position,               # 3 values
            normalized_velocity,               # 3 values
            projected_vel,                     # 1 value
            normalized_speed,                  # 1 value
            normalized_delta_velocity,         # 3 values
            normalized_angular_speed,          # 1 value
            normalized_angular_velocity,       # 3 values
            one_hot,                          # num_targets values
            sin_yaw,                          # 1 value
            cos_yaw,                          # 1 value
            gate_ratios,                      # 1 * (n_opponents) values
            normalized_time,                  # 1 value
            normalized_heading_error,         # 1 value
            normalized_progress,              # 1 value
            all_opponent_rel_pos,            # 3 * (n_opponents) values
            all_opponent_distances,          # 1 * (n_opponents) values
            all_opponent_velocities,         # 3 * (n_opponents) values
            all_opponent_progresses          # 1 * (n_opponents) values
        ])


        self.vehicles[agent]["prev_velocity"] = np.copy(velocity)

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