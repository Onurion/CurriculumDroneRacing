import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from collections import deque


class DroneRaceCurriculumEnv(gym.Env):
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
        super(DroneRaceCurriculumEnv, self).__init__()

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

        self.is_buffer_obs = is_buffer_obs

        # Define two drones (agents) for self-play.
        self.agents = ["drone0", "drone1"]
        self.n_agents = len(self.agents)

        # Create a dictionary to hold each agent's components and info.
        self.vehicles = {}
        for agent in self.agents:
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
            self.vehicles[agent] = {
                "drone": drone,
                "low_level_controller": low_level_controller,
                "vel_tracker": vel_tracker,
                "current_target_index": 0,   # Which gate is the current target.
                "prev_distance": None,       # Last recorded distance to the current target.
                "prev_velocity": None,       # Last recorded velocity norm.
                "progress": 0,               # Number of gates passed.
                "prev_progress": 0           # For checking overtaking progress.
            }

        self.n_gates = int(n_gates)
        # Define the gates. Here, gates are arranged in a circle.
        angles = np.linspace(0, 2 * np.pi, self.n_gates + 1)[:-1]

        self.gates = []
        for idx, theta in enumerate(angles):
            if idx % 2 == 0:
                z = 4 #np.random.uniform(3.5, 4.5)  # high altitude range
            else:
                z = 3 #np.random.uniform(1.5, 2.5)  # low altitude range
            center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
            gate = {"center": center, "yaw": theta + np.pi/2}
            self.gates.append(gate)
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])


        # Thresholds and reward shaping parameters.
        self.target_threshold = 0.5

        self.max_distance = 10.0
        self.max_vel = 10.0

        # Observation: the observation vector length is 17 + num_targets.
        self.num_targets = self.gate_positions.shape[0]

        # Reward function selector.
        self.reward_function = self.reward_function_1 if reward_type == 1 else self.reward_function_2
        if observation_type == 1:
            self.obs_dim = 17 + self.num_targets
            self.observation_function = self._get_obs_1
        else:
            self.obs_dim = 33 + self.num_targets
            self.observation_function = self._get_obs_2


        if self.is_buffer_obs:
            self.buffer_size = buffer_size  # This is N (number of timesteps)
            self.full_obs_dim = self.obs_dim * self.buffer_size
            # Initialize the deque with a fixed maximum length
            self.obs_buffer = {
                agent: deque(maxlen=buffer_size) for agent in self.agents
            }

            for agent in self.agents:
                for _ in range(self.buffer_size):
                    self.obs_buffer[agent].append(np.zeros(self.obs_dim, dtype=np.float32))
        else:
            self.full_obs_dim = self.obs_dim  # no buffering, full_obs_dim equals

        # Define observation and action spaces per agent.
        self.observation_space = gym.spaces.Dict({
            "drone0": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.full_obs_dim,), dtype=np.float32),
            "drone1": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.full_obs_dim,), dtype=np.float32)
        })
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = {agent: single_action_space for agent in self.agents}

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

    def update_curriculum_params(self, params: dict):
        """
        Update environment parameters based on the curriculum stage.
        For example, params can include: 'action_coefficient', 'minimum_velocity',
        'enable_collision', 'collision_distance', 'enable_takeover', 'takeover_reward'
        """
        for key, value in params.items():
            setattr(self, key, value)
        print("Curriculum parameters updated:", params)

    def reset(self, *, seed=None, return_info=False, options=None):
        self.current_step = 0
        obs = {}

        # If buffering observations, clear and prefill the per-agent buffer.
        if self.is_buffer_obs:
            for agent in self.agents:
                # Clear the buffer.
                self.obs_buffer[agent].clear()

                # Prefill the buffer with zeros (or an initial observation if available).
                for _ in range(self.buffer_size):
                    self.obs_buffer[agent].append(
                        np.zeros(self.obs_dim, dtype=np.float32)
                    )

        offset_distance = np.random.uniform(low=1.5, high=3.0)
        jitter_range = np.random.uniform(low=self.jitter_range[0], high=self.jitter_range[1])
        gate_index = np.random.randint(low=0, high=len(self.gate_positions))

        for agent in self.agents:
            # Randomly initialize the drone's position.
            # Use this version:
            if self.random_init:
                position = self.sample_safe_start_position()
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

            current_obs = self.observation_function(agent)

            if self.is_buffer_obs:
                # Append the latest observation.
                self.obs_buffer[agent].append(current_obs)

                # Option 1: Concatenate to form a single flattened vector.
                obs[agent] = np.concatenate(list(self.obs_buffer[agent]), axis=0)

                # Option 2 (alternative): Stack observations into a 2D tensor.
                # obs[agent] = np.stack(self.obs_buffer[agent], axis=0)
            else:
                obs[agent] = current_obs


            obs[agent] = np.clip(
                obs[agent],
                self.observation_space[agent].low,
                self.observation_space[agent].high
            ).astype(np.float32)

        return obs, {}

    def step(self, actions):
        rewards = {}
        obs = {}
        dones = {}
        infos = {}

        # Increment the global step count.
        self.current_step += 1

        # Process each drone's action.
        for agent in self.agents:
            # Get agent-specific action and vehicle data.

            act = actions[agent]
            vehicle = self.vehicles[agent]
            act = np.clip(act, self.action_space[agent].low, self.action_space[agent].high)
            act = self.action_coefficient * act  # Scale action based on the curriculum.

            drone = vehicle["drone"]
            state = drone.state
            velocity = state[3:6]
            vel_norm = np.linalg.norm(velocity)
            desired_velocity = velocity + act * self.dt

            # Compute control command.
            control = vehicle["vel_tracker"].compute_control(np.array(desired_velocity))
            new_state, _ = drone.step(control)

            # Update state variables.
            position = new_state[:3]
            velocity = new_state[3:6]
            roll, pitch, yaw = new_state[6], new_state[7], new_state[8]

            # Retrieve current target information.
            current_target_idx = vehicle["current_target_index"]
            current_gate_position = self.gate_positions[current_target_idx]
            current_gate_yaw = self.gate_yaws[current_target_idx]

            # Compute relative position data.
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
            dz = position[2] - current_gate_position[2]

            cos_yaw = np.cos(-current_gate_yaw)
            sin_yaw = np.sin(-current_gate_yaw)
            rx = cos_yaw * dx - sin_yaw * dy
            ry = sin_yaw * dx + cos_yaw * dy
            # Here, we assume that the gate is upright, so the vertical (z) axis is already aligned.
            rz = dz
            in_xy = (abs(rx) <= self.gate_passing_tolerance) and (abs(ry) <= self.gate_passing_tolerance)
            z_diff = abs(position[2] - current_gate_position[2])
            in_z = (z_diff <= self.gate_passing_tolerance)
            target_reached = in_xy and in_z

            if target_reached:
                reward = 1.0
                # Move to the next gate (cyclic order) and increment progress.
                vehicle["current_target_index"] = (vehicle["current_target_index"] + 1) % self.num_targets
                vehicle["progress"] += 1
                target_reached = False


            # Check out-of-bound condition.
            all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
            done = False
            if np.min(all_distances) > self.max_distance:
                done = True

            infos[agent] = {"progress": vehicle["progress"], "velocity": vel_norm, "collision": 0}
            # Check for excessive roll or pitch.
            if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
                done = True
                reward = -1.0
                infos[agent]["collision"] = 1

            # Save the current distance for reward shaping on the next step.
            vehicle["prev_distance"] = distance

            rewards[agent] = reward
            dones[agent] = done


        # --- Collision Penalty (Curriculum Controlled) ---
        if self.enable_collision:
            pos0 = self.vehicles["drone0"]["drone"].state[:3]
            pos1 = self.vehicles["drone1"]["drone"].state[:3]
            if np.linalg.norm(pos0 - pos1) < self.drone_collision_margin:
                for agent in self.agents:
                    rewards[agent] -= self.collision_penalty  # Penalty for collision.
                    infos[agent]["collision"] = 1
                if self.terminate_on_collision:
                    dones["__all__"] = True

        # --- Overtaking Bonus (Curriculum Controlled) ---
        if self.enable_takeover:
            progress0 = self.vehicles["drone0"]["progress"]
            progress1 = self.vehicles["drone1"]["progress"]
            if progress0 > progress1:
                rewards["drone0"] += self.takeover_reward
            elif progress1 > progress0:
                rewards["drone1"] += self.takeover_reward

        # Clip rewards and update previous progress.
        for agent in self.agents:
            rewards[agent] = np.clip(rewards[agent], -1.0, 1.0)
            self.vehicles[agent]["prev_progress"] = self.vehicles[agent]["progress"]

        # Set global termination flag (e.g., when max steps reached).
        global_done = self.current_step >= self.max_steps
        dones["__all__"] = global_done
        truncated = global_done

        # Get updated observations.
        for agent in self.agents:
            current_obs = self.observation_function(agent)  # Shape: (obs_dim,)
            if self.is_buffer_obs:
                self.obs_buffer[agent].append(current_obs)
                obs[agent] = np.concatenate(list(self.obs_buffer[agent]), axis=0)
            else:
                obs[agent] = current_obs

            obs[agent] = np.clip(
                obs[agent],
                self.observation_space[agent].low,
                self.observation_space[agent].high
            ).astype(np.float32)

        return obs, rewards, dones, truncated, infos

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



        # Determine opponent key.
        opponent = "drone1" if agent == "drone0" else "drone0"
        opponent_position = self.vehicles[opponent]["drone"].state[0:3]
        opponent_rel_pos = opponent_position - position
        normalized_opponent_rel_pos = opponent_rel_pos / self.max_distance
        opponent_distance = np.linalg.norm(opponent_rel_pos)
        normalized_opponent_distance = np.array([opponent_distance / self.max_distance], dtype=np.float32)

        opponent_velocity = self.vehicles[opponent]["drone"].state[3:6]
        normalized_opponent_velocity = opponent_velocity / self.max_vel

        gate_ratio = (self.vehicles[agent]["progress"] + 1e-6) / (self.vehicles[opponent]["progress"] + 1e-6)
        gate_ratio = np.array([gate_ratio], dtype=np.float32)

        normalized_opponent_progress = np.array([self.vehicles[opponent]["progress"] / 25.0], dtype=np.float32)

        delta_velocity = velocity - self.vehicles[agent]["prev_velocity"]  # or compute as (current_velocity - previous_velocity) if

        normalized_delta_velocity = delta_velocity / self.max_vel

        normalized_time = np.array([self.current_step / self.max_steps], dtype=np.float32)


        obs = np.concatenate([
            normalized_relative_position,      # 3 values.
            normalized_distance,               # 1 value.
            normalized_position,               # 3 values.
            normalized_velocity,               # 3 values.
            projected_vel,                     # 1 value.
            normalized_speed,                  # 1 value.
            normalized_delta_velocity,         # 3 values.
            normalized_angular_speed,          # 1 value.
            normalized_angular_velocity,       # 3 values.
            one_hot,                           # num_targets values.
            sin_yaw,                           # 1 value.
            cos_yaw,                           # 1 value.
            gate_ratio,                        # 1 value.
            normalized_time,                   # 1 value.
            normalized_heading_error,          # 1 value.
            normalized_progress,               # 1 value.
            normalized_opponent_rel_pos,       # 3 values.
            normalized_opponent_distance,      # 1 value.
            normalized_opponent_velocity,      # 3 values.
            normalized_opponent_progress       # 1 value.
        ])

        self.vehicles[agent]["prev_velocity"] = np.copy(velocity)

        return obs

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

        # Determine opponent key
        opponent = "drone1" if agent == "drone0" else "drone0"
        opponent_position = self.vehicles[opponent]["drone"].state[0:3]
        # Compute relative opponent position
        opponent_rel_pos = opponent_position - position
        normalized_opponent_rel_pos = opponent_rel_pos / self.max_distance
        opponent_distance = np.linalg.norm(opponent_rel_pos)
        normalized_opponent_distance = np.array([opponent_distance / self.max_distance], dtype=np.float32)

        obs = np.concatenate([
            norm_rel_pos,         # 3 values
            norm_distance,        # 1 value
            norm_velocity,        # 3 values
            projected_vel,        # 1 value
            one_hot,              # num_targets values
            norm_position,        # 3 values
            sin_yaw,              # 1 value
            cos_yaw,              # 1 value
            normalized_opponent_rel_pos,  # 3 values: relative opponent position normalized
            normalized_opponent_distance  # 1 value: normalized opponent distance
        ])

        return obs

    def render(self, mode='human'):
        for agent in self.agents:
            pos = self.vehicles[agent]["drone"].state[:3]
            current_target = self.gate_positions[self.vehicles[agent]["current_target_index"]]
            distance = np.linalg.norm(current_target - pos)
            print(f"Agent: {agent} | Dist: {distance:.2f} | Target Index: {self.vehicles[agent]['current_target_index']} | "
                  f"Target: {current_target} | Step: {self.current_step} | Position: {pos} ")

    def get_state(self, agent):
        return self.vehicles[agent]["drone"].state



