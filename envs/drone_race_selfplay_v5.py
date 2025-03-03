import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker

class DroneRaceSelfPlayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_gates: int = 5, radius=10, min_vel=0.2, action_coeff=1.0, distance_exp_decay=2.0, w_distance= 0.0, w_distance_change=1.0, w_alignment=0.5, w_deviation=0.0, w_inactivity=0.0, reward_type=2):
        super(DroneRaceSelfPlayEnv, self).__init__()

        # Simulation parameters.
        self.dt = 0.1
        self.max_steps = 2000
        self.current_step = 0
        self.gate_size = 1.0  # 1m x 1m gate
        self.passing_tolerance = self.gate_size / 2.0
        self.action_coeff = action_coeff
        self.reward_function = self.reward_function_1 if reward_type == 1 else self.reward_function_2

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
        self.min_active_vel = min_vel

        self.max_distance = 10.0
        self.max_vel = 10.0

        # Observation: same as before, the observation vector length is 13 + num_targets.
        self.obs_dim = 17 + self.num_targets

        # Define observation and action spaces per agent.
        # single_obs_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # self.observation_space = spaces.Dict({agent: single_obs_space for agent in self.agents})
        # single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # self.action_space = spaces.Dict({agent: single_action_space for agent in self.agents})
        # For each agent.
        # self.observation_space = {agent: single_obs_space for agent in self.agents}
        self.observation_space = gym.spaces.Dict({
            "drone0": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32),
            "drone1": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        })
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = {agent: single_action_space for agent in self.agents}

        # Set collision threshold between drones.
        self.collision_distance = 0.5  # in meters

        # Reward parameters.
        self.a = distance_exp_decay                   # Controls the shape of the exponential decay.
        self.w_distance = w_distance             # Weight for base_reward.
        self.w_distance_change = w_distance_change             # Weight for bonus_distance.
        self.w_deviation = w_deviation            # Weight for deviation penalty.
        self.w_inactivity = w_inactivity       # Weight for inactivity penalty.
        self.w_alignment = w_alignment


    def reset(self, *, seed=None, return_info=False, options=None):
        self.current_step = 0
        obs_dict = {}
        for agent in self.agents:
            # Randomly initialize the drone's position.
            position = np.random.uniform(low=-self.max_distance, high=self.max_distance, size=3).astype(np.float32)
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
            vehicle["progress"] = 0
            vehicle["prev_progress"] = 0

            obs_dict[agent] = self._get_obs_agent(agent)
        return obs_dict, {}

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
            act = self.action_coeff * act  # scale as in your original design

            drone = vehicle["drone"]
            state = drone.state
            velocity = state[3:6]
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

            reward = self.reward_function(distance, vehicle["prev_distance"], velocity, unit_rel_pos)

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
                # Move to the next gate (cyclic order) and increment progress.
                vehicle["current_target_index"] = (vehicle["current_target_index"] + 1) % self.num_targets
                vehicle["progress"] += 1
                target_reached = False

            # Check out-of-bound condition.
            all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
            done = False
            if np.min(all_distances) > self.max_distance:
                done = True

            # Check for excessive roll or pitch.
            if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
                done = True
                reward = -1.0

            # Save the current distance for reward shaping on the next step.
            vehicle["prev_distance"] = distance

            rewards[agent] = reward
            dones[agent] = done
            infos[agent] = {"progress": vehicle["progress"], "position": position}

        # # --- Collision Penalty ---
        # pos0 = self.vehicles["drone0"]["drone"].state[:3]
        # pos1 = self.vehicles["drone1"]["drone"].state[:3]
        # if np.linalg.norm(pos0 - pos1) < self.collision_distance:
        #     for agent in self.agents:
        #         rewards[agent] -= 0.5  # Penalty for collision

        # --- Overtaking Bonus ---
        # progress0 = self.vehicles["drone0"]["progress"]
        # progress1 = self.vehicles["drone1"]["progress"]
        # if progress0 > progress1:
        #     rewards["drone0"] += 0.2
        # elif progress1 > progress0:
        #     rewards["drone1"] += 0.2

        # Clip rewards again.
        for agent in self.agents:
            rewards[agent] = np.clip(rewards[agent], -1.0, 1.0)
            # Update previous progress (can be used if you later want to track delta gains)
            self.vehicles[agent]["prev_progress"] = self.vehicles[agent]["progress"]

        # Set global termination flag (for example, when max steps reached).
        global_done = self.current_step >= self.max_steps
        dones["__all__"] = global_done
        truncated = global_done

        # Get updated observations.
        for agent in self.agents:
            obs[agent] = self._get_obs_agent(agent)

        return obs, rewards, dones, truncated, infos

    def reward_function_2(self, distance, prev_distance, velocity, unit_rel_pos):

        x = distance / self.max_distance
        base_reward = 2 * (np.exp(-self.a * x) - np.exp(-self.a)) / (1 - np.exp(-self.a)) - 1
        delta_distance = prev_distance - distance
        bonus_distance = 2.5 * delta_distance
        vel_norm = np.linalg.norm(velocity)

        # Compute the alignment component
        vel_norm = np.linalg.norm(velocity)

        cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
        alignment_component = np.abs(vel_norm - self.min_active_vel) * cos_angle

        inactivity_penalty = min(vel_norm - self.min_active_vel, 0)

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

        inactivity_penalty = 0.0
        if vel_norm < self.min_active_vel:
            inactivity_penalty = (self.min_active_vel - vel_norm)

        if vel_norm > 1e-6:
            cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
            deviation = 1.0 - cos_angle
        else:
            deviation = 0.0

        # Combine the reward components.
        reward = (self.w_distance * base_reward +
                  self.w_distance_change * bonus_distance -
                  self.w_deviation * deviation -
                  self.w_inactivity * inactivity_penalty)


        return reward

    def _get_obs_agent(self, agent):
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

        obs = np.clip(obs, self.observation_space[agent].low, self.observation_space[agent].high).astype(np.float32)
        return obs

    def render(self, mode='human'):
        for agent in self.agents:
            pos = self.vehicles[agent]["drone"].state[:3]
            current_target = self.gate_positions[self.vehicles[agent]["current_target_index"]]
            distance = np.linalg.norm(current_target - pos)
            print(f"Agent: {agent} | Dist: {distance:.2f} |  Target Index: {self.vehicles[agent]['current_target_index']} | "
                  f"Target: {current_target} | Step: {self.current_step} | Position: {pos} ")

    def get_state(self, agent):
        return self.vehicles[agent]["drone"].state

