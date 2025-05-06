import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker
import numpy as np
from collections import deque


class DroneRaceCurriculumMultiEnv(gym.Env):
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
                 evaluation_mode: bool=False,
                 track_type: str="circle",
                 # Curriculum parameters:
                 enable_collision: bool=False,
                 terminate_on_collision: bool=False,
                 collision_penalty: float = 0.5,
                 drone_collision_margin: float =0.5,
                 gate_passing_tolerance: float = 0.5,
                 enable_overtake:bool=False,
                 overtake_reward: float =0.0,
                 jitter_range:list = [0.25, 0.75],
                 **kwargs):
        super(DroneRaceCurriculumMultiEnv, self).__init__()

        # Simulation parameters.
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        self.gate_size = gate_size  # 1m x 1m gate
        self.gate_passing_tolerance = gate_passing_tolerance
        self.random_init = random_init
        self.jitter_range = jitter_range
        self.evaluation_mode = evaluation_mode
        self.track_type = track_type
        # self.eval_scenarios = ["overtaking", "tight", "overtaking", "approach", "normal"]
        self.eval_scenarios = ["normal"]

        # Curriculum parameters.
        self.action_coefficient = action_coefficient          # Multiplier applied to the raw action.
        self.minimum_velocity = minimum_velocity     # Minimum velocity (used to penalize inactivity).
        self.enable_collision = enable_collision  # Whether collision detection is enabled.
        self.collision_penalty = collision_penalty  # Penalty for collision.
        self.drone_collision_margin = drone_collision_margin  # Drone Collision threshold in meters.
        self.terminate_on_collision = terminate_on_collision  # Terminate episode on collision.
        self.enable_overtake = enable_overtake    # Whether overtake bonus is enabled.
        self.overtake_reward = overtake_reward    # Reward bonus for overtaking in selfplay.
        

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
        self.training_agent = "drone0"
        self.n_agents = n_agents
        # Define two drones (agents) for self-play.
        for i in range(self.n_agents):
            self.agents.append(f"drone{i}")


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
                "prev_progress": 0,           # For checking overtaking progress.
                "reward": 0
            }

        self.n_gates = int(n_gates)
        # # Define the gates. Here, gates are arranged in a circle.
        # angles = np.linspace(0, 2 * np.pi, self.n_gates + 1)[:-1]

        # self.gates = []
        # for idx, theta in enumerate(angles):
        #     if idx % 2 == 0:
        #         z = 4 # np.random.uniform(3.5, 4.5)  # high altitude range
        #     else:
        #         z = 3 # np.random.uniform(1.5, 2.5)  # low altitude range
        #     center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
        #     gate = {"center": center, "yaw": theta + np.pi/2}
        #     self.gates.append(gate)
        # self.gate_positions = np.array([gate["center"] for gate in self.gates])
        # self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])

        if self.track_type == "circle":
            self.setup_gates(n_gates, radius)
        elif self.track_type == "infinity":
            self.setup_gates_v2()
        else:   
            raise("Invalid track type.")
            


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

        obs_dict = {}
        for agent in self.agents:
            obs_dict[agent] = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.full_obs_dim,), dtype=np.float32)
        # Define observation and action spaces per agent.
        self.observation_space = gym.spaces.Dict(obs_dict)
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = {agent: single_action_space for agent in self.agents}

        self.prev_state_info = {}

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

    def setup_gates(self, n_gates=5, radius=10):
        self.n_gates = int(n_gates)
        self.gate_width = 2.0  # Define your gate width
        self.gate_height = 2.0  # Define your gate height
        
        # Define the gates arranged in a circle
        angles = np.linspace(0, 2 * np.pi, self.n_gates + 1)[:-1]
        
        self.gates = []
        for idx, theta in enumerate(angles):
            if idx % 2 == 0:
                z = 4  # high altitude
            else:
                z = 3  # low altitude
                
            center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float32)
            
            # The yaw is perpendicular to the radial direction
            # This makes the gate face toward the center of the circle
            yaw = theta + np.pi/2
            
            gate = {
                "center": center, 
                "yaw": yaw,
                "width": self.gate_width,
                "height": self.gate_height
            }
            self.gates.append(gate)
        
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])

    def setup_gates_v2(self):
        self.n_gates = 6
        self.gate_width = 2.0
        self.gate_height = 2.0
        

        gate_positions = [
            [5.0, -5, 1],      # Gate 1 (right)
            [10.0, 0, 2],       # Gate 2 (back right)
            [5.0, 5, 2],       # Gate 3 (back)
            [-5.0, -5, 0.5],  # Gate 4 (middle)
            [-10.0, 0, 0.5],   # Gate 5 (front left)
            [-5.0, 5, 1.5]    # Gate 6 (left)
        ]
        
        # Gate orientations (yaw angles in radians)
        gate_yaws = [
            np.pi/2,        # Gate 1 facing +y
            0,              # Gate 2 facing +x
            -np.pi/2,       # Gate 3 facing -y
            -np.pi/2,       # Gate 4 facing -y
            0,              # Gate 5 facing +x
            np.pi/2         # Gate 6 facing +y
        ]
        
        self.gates = []
        for idx in range(self.n_gates):
            center = np.array(gate_positions[idx], dtype=np.float32)
            yaw = gate_yaws[idx]
            
            gate = {
                "center": center,
                "yaw": yaw + np.pi/2,  # Adjust to face the gate
                "width": self.gate_width,
                "height": self.gate_height
            }
            self.gates.append(gate)
        
        self.gate_positions = np.array([gate["center"] for gate in self.gates])
        self.gate_yaws = np.array([gate["yaw"] for gate in self.gates])


    def reset(self, *, seed=None, return_info=False, options=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            return_info: Whether to return info dict
            options: Optional configuration overrides
            
        Returns:
            obs: Initial observations for all agents
            info: Empty dict (if return_info is True)
        """
        self.current_step = 0
        obs = {}
        self.prev_state_info = {}
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
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

        # Select a random gate for potential start positioning
        gate_index = np.random.randint(low=0, high=len(self.gate_positions))
        
        # Get agent positions using strategy-based initialization
        agent_positions = self._initialize_agent_positions(gate_index)
        
        # Check and ensure positions have sufficient separation if needed
        if not self.evaluation_mode and len(self.agents) > 1:
            agent_positions = self._ensure_safe_separation(agent_positions)
        
        # Initialize all agents with their positions
        for agent_idx, agent in enumerate(self.agents):
            position = agent_positions[agent]
            
            # Ensure minimum height
            position[2] = max(0.5, position[2])
            
            # Reset vehicle state
            velocity = np.zeros(3, dtype=np.float32)
            vehicle = self.vehicles[agent]
            vehicle["drone"].reset()
            vehicle["vel_tracker"].prev_error = np.zeros(3, dtype=np.float32)
            vehicle["drone"].state[0:3] = np.copy(position)
            vehicle["drone"].state[3:6] = np.copy(velocity)

            # Determine the nearest gate as the first target
            distances = np.linalg.norm(self.gate_positions - position, axis=1)
            vehicle["current_target_index"] = int(np.argmin(distances))
            vehicle["distance_to_gate"] = distances[vehicle["current_target_index"]]
            vehicle["prev_distance"] = distances[vehicle["current_target_index"]]
            vehicle["prev_velocity"] = np.zeros(3, dtype=np.float32)
            vehicle["progress"] = 0
            vehicle["prev_progress"] = 0
            vehicle["reward"] = 0

            # Get initial observation
            current_obs = self.observation_function(agent)

            if self.is_buffer_obs:
                # Append the latest observation.
                self.obs_buffer[agent].append(current_obs)
                obs[agent] = np.concatenate(list(self.obs_buffer[agent]), axis=0)
            else:
                obs[agent] = current_obs

            obs[agent] = np.clip(
                obs[agent],
                self.observation_space[agent].low,
                self.observation_space[agent].high
            ).astype(np.float32)

        return obs, {}

    def _initialize_agent_positions(self, gate_index):
        """
        Initialize positions for all agents according to current strategy.
        
        Args:
            gate_index: Index of a selected gate for potential start positioning
            
        Returns:
            agent_positions: Dictionary mapping agent IDs to initial positions
        """
        agent_positions = {}
        
        # Determine if we should use close initialization (either in evaluation or with probability)
        use_close_init = (not self.random_init or 
                        (hasattr(self, 'close_start_probability') and 
                        np.random.random() < self.close_start_probability))
        
        # ---- EVALUATION MODE ----
        if self.evaluation_mode:
            # Select a scenario if in evaluation mode
            if hasattr(self, 'eval_scenarios') and self.eval_scenarios:
                eval_type = np.random.choice(self.eval_scenarios)
            else:
                eval_type = 'normal'
                
            # Use specially configured positions for each agent
            for idx, agent in enumerate(self.agents):
                # Store agent index for positioning
                self.drone_id = idx
                self.num_drones = len(self.agents)
                
                # Get position using evaluation positioning
                agent_positions[agent] = self.initialize_drones_closer(evaluation_type=eval_type)
        
        # ---- TRAINING MODE ----
        else:
            if use_close_init:
                # Use closer initialization occasionally during training
                offset_distance = np.random.uniform(low=1.5, high=3.0)
                jitter_range = np.random.uniform(
                    low=self.jitter_range[0], 
                    high=self.jitter_range[1]
                )
                
                for idx, agent in enumerate(self.agents):
                    # Store agent index for staggered positioning
                    self.drone_id = idx
                    self.num_drones = len(self.agents)
                    
                    # Position drones with appropriate spacing
                    agent_positions[agent] = self.initialize_drones_closer(
                        current_gate=gate_index,
                        offset_distance=offset_distance,
                        jitter_range=jitter_range,
                        evaluation_type='normal'  # Use normal spacing during training
                    )
            else:
                # Use strategy-based random initialization
                init_strategy = getattr(self, 'init_strategy', 'uniform')
                
                for agent in self.agents:
                    agent_positions[agent] = self.sample_safe_start_position(strategy=init_strategy)
        
        return agent_positions

    def _ensure_safe_separation(self, positions):
        """
        Ensure all drone positions have safe separation.
        
        Args:
            positions: Dictionary of agent->position mappings
            
        Returns:
            Updated positions with safe separation
        """
        # Define minimum safe distance between drones
        min_distance = self.drone_collision_margin * 3.0  # At least 3x the collision margin
        max_attempts = 50
        
        # Convert to list of (agent, position) pairs for easier manipulation
        position_list = list(positions.items())
        
        # Try to adjust positions to ensure safe separation
        for _ in range(max_attempts):
            need_adjustment = False
            
            # Check all pairs of drones
            for i in range(len(position_list)):
                for j in range(i+1, len(position_list)):
                    agent_i, pos_i = position_list[i]
                    agent_j, pos_j = position_list[j]
                    
                    # Calculate distance between drones
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # If too close, mark for adjustment
                    if distance < min_distance:
                        need_adjustment = True
                        
                        # Adjust the position of the second drone
                        direction = pos_j - pos_i
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        else:
                            direction = np.array([1, 0, 0])  # Default direction if same position
                        
                        # Move the second drone away from the first
                        new_pos = pos_j + direction * (min_distance - distance + 0.1)
                        
                        # Ensure within bounds
                        new_pos[0] = np.clip(new_pos[0], -self.max_distance, self.max_distance)
                        new_pos[1] = np.clip(new_pos[1], -self.max_distance, self.max_distance)
                        new_pos[2] = max(0.5, new_pos[2])
                        
                        # Update position in the list
                        position_list[j] = (agent_j, new_pos)
            
            # If no adjustments needed, we're done
            if not need_adjustment:
                break
        
        # Convert back to dictionary
        updated_positions = {agent: pos for agent, pos in position_list}
        return updated_positions

    def sample_safe_start_position(self, strategy='uniform'):
        """
        Sample diverse starting positions for training to encourage exploration.
        
        Args:
            strategy: Initialization strategy to use
                - 'uniform': Completely random positions
                - 'near_gate': Positions near gates
                - 'far_gate': Positions far from gates
                - 'approach_angle': Positions along valid approach angles
                - 'mixed': Mix of different strategies
                - 'varied': More varied positions with different heights
                - 'competitive': Positions that mimic race starts
        
        Returns:
            position: Safe 3D starting position for a drone
        """
        # If strategy is mixed, randomly select a sub-strategy
        if strategy == 'mixed':
            strategy = np.random.choice(['uniform', 'near_gate', 'approach_angle'], 
                                    p=[0.5, 0.3, 0.2])
        elif strategy == 'varied':
            strategy = np.random.choice(['uniform', 'near_gate', 'far_gate', 'approach_angle'], 
                                    p=[0.3, 0.3, 0.2, 0.2])
        elif strategy == 'competitive':
            strategy = np.random.choice(['approach_angle', 'near_gate', 'tight_group'], 
                                    p=[0.5, 0.3, 0.2])
        
        # Define bounds
        low = np.array([-self.max_distance, -self.max_distance, 0.5])  # Minimum height of 0.5
        high = np.array([self.max_distance, self.max_distance, self.max_distance])
        safe_distance = self.gate_size * 1.5  # Safety margin
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if strategy == 'uniform':
                # Completely random position
                position = np.random.uniform(low=low, high=high).astype(np.float32)
                
            elif strategy == 'near_gate':
                # Position near a random gate (but not too close)
                gate_idx = np.random.randint(0, len(self.gates))
                gate_pos = self.gates[gate_idx]['center'] 
                
                # Random direction from gate
                direction = np.random.uniform(-1, 1, size=3)
                direction = direction / np.linalg.norm(direction)
                
                # Random distance from gate (somewhat close)
                distance = np.random.uniform(safe_distance, safe_distance*3)
                position = gate_pos + direction * distance
                position[2] = max(0.5, position[2])  # Ensure minimum height
                
            elif strategy == 'far_gate':
                # Position far from gates for long-distance navigation
                gate_idx = np.random.randint(0, len(self.gates))
                gate_pos = self.gates[gate_idx]['center']
                
                # Random direction from gate
                direction = np.random.uniform(-1, 1, size=3)
                direction = direction / np.linalg.norm(direction)
                
                # Far distance from gate
                distance = np.random.uniform(safe_distance*3, self.max_distance*0.7)
                position = gate_pos + direction * distance
                position[2] = max(0.5, position[2])  # Ensure minimum height
                
            elif strategy == 'approach_angle':
                # Position along a valid approach angle to gate
                gate_idx = np.random.randint(0, len(self.gates))
                gate = self.gates[gate_idx]
                gate_pos = gate['center']
                gate_yaw = gate['yaw']
                
                # Generate position in front or behind gate (along gate normal)
                front_side = np.random.choice([True, False])
                direction = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0])
                if not front_side:
                    direction = -direction
                    
                # Add some perpendicular drift for varied approaches
                perp_direction = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0])
                perp_amount = np.random.uniform(-2.0, 2.0)
                
                # Add some height variation
                height_offset = np.random.uniform(-1.0, 1.0)
                height_dir = np.array([0, 0, 1])
                
                # Generate position
                distance = np.random.uniform(safe_distance, self.max_distance*0.6)
                position = (gate_pos + 
                        direction * distance + 
                        perp_direction * perp_amount +
                        height_dir * height_offset)
                position[2] = max(0.5, position[2])  # Ensure minimum height
                
            elif strategy == 'tight_group':
                # Create positions as if in a tight race start
                gate_idx = np.random.randint(0, len(self.gates))
                gate = self.gates[gate_idx]
                gate_pos = gate['center']
                gate_yaw = gate['yaw']
                
                # Position behind gate
                direction = -np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0])
                offset_distance = np.random.uniform(1.5, 3.0)
                
                # Add small random offset
                lateral_offset = np.random.uniform(-0.5, 0.5)
                perp_direction = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0])
                
                position = gate_pos + direction * offset_distance + perp_direction * lateral_offset
                position[2] = max(0.5, gate_pos[2] + np.random.uniform(-0.3, 0.3))
            
            # Check if position is safe (not too close to any gate)
            safe = True
            for gate in self.gates:
                gate_pos = gate['center']
                distance = np.linalg.norm(position - gate_pos)
                if distance < safe_distance:
                    safe = False
                    break
            
            # Enforce domain boundaries
            if (position[0] < low[0] or position[0] > high[0] or
                position[1] < low[1] or position[1] > high[1] or
                position[2] < low[2] or position[2] > high[2]):
                safe = False
                
            if safe:
                return position
        
        # Fallback to a simple safe position if all attempts fail
        return np.array([0, 0, self.gates[0]['center'][2] + 2.0], dtype=np.float32)

    # def initialize_drones_closer(self, current_gate=0, offset_distance=2.0, jitter_range=0.5, evaluation_type='normal'):
    #     """
    #     Initialize drones for evaluation or controlled starts.
        
    #     Args:
    #         current_gate: Gate index to position drones near
    #         offset_distance: Base distance from gate
    #         jitter_range: Amount of random variation to apply
    #         evaluation_type: Type of evaluation scenario:
    #             - 'normal': Standard race start
    #             - 'tight': Very close positioning for collision avoidance testing
    #             - 'overtaking': Setup to test overtaking behavior
    #             - 'approach': Test gate approach precision
        
    #     Returns:
    #         position: 3D position for drone
    #     """
    #     # Select the gate
    #     gate = self.gates[current_gate]
    #     gate_center = gate['center']
    #     gate_yaw = gate['yaw']
        
    #     # Base direction vectors
    #     forward_dir = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
    #     right_dir = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
    #     up_dir = np.array([0.0, 0.0, 1.0])

    #     # Set parameters based on evaluation type
    #     if evaluation_type == 'normal':
    #         # Standard race start - reasonable distance in front of gate
    #         base_distance = offset_distance
    #         lateral_range = jitter_range  # How far drones spread laterally
    #         height_range = jitter_range * 0.6   # How much height variation
            
    #     elif evaluation_type == 'tight':
    #         # Very close positioning to test collision avoidance
    #         base_distance = offset_distance * 0.5
    #         lateral_range = jitter_range * 0.4  # Tighter lateral spacing
    #         height_range = jitter_range * 0.3   # Minimal height variation
            
    #     elif evaluation_type == 'overtaking':
    #         # Positioning that encourages overtaking scenarios
    #         base_distance = offset_distance * 1.5
    #         lateral_range = jitter_range * 0.6  # Moderate lateral spacing
    #         height_range = jitter_range * 0.4   # Some height variation
            
    #     elif evaluation_type == 'approach':
    #         # Test precise gate approach
    #         base_distance = offset_distance * 2.0
    #         lateral_range = jitter_range * 1.5  # Wider lateral spacing
    #         height_range = jitter_range  # More height variation to test approach angles
    #     else:
    #         # Default to normal
    #         base_distance = offset_distance
    #         lateral_range = jitter_range
    #         height_range = jitter_range * 0.6
        
    #     # Compute base position before gate
    #     base_position = gate_center - forward_dir * base_distance
        
    #     # Add lateral offset based on drone ID or random
    #     # This creates a line of drones before the gate
    #     if hasattr(self, 'drone_id') and hasattr(self, 'num_drones'):
    #         # If we know which drone this is out of how many
    #         drone_id = self.drone_id
    #         num_drones = self.num_drones
            
    #         # Spread drones evenly
    #         if num_drones > 1:
    #             lateral_offset = (drone_id - (num_drones-1)/2) * lateral_range
                
    #             # For overtaking scenario, stagger drones forward/backward too
    #             if evaluation_type == 'overtaking':
    #                 forward_offset = (drone_id % 2) * 1.0  # Alternate drones forward/back
    #                 base_position = base_position + forward_dir * forward_offset
    #         else:
    #             lateral_offset = 0
    #     else:
    #         # If we don't have drone ID, use random offset
    #         lateral_offset = np.random.uniform(-lateral_range, lateral_range)
        
    #     # Apply lateral and height offsets
    #     height_offset = np.random.uniform(-height_range, height_range)
    #     position = (base_position + 
    #             right_dir * lateral_offset + 
    #             up_dir * height_offset)
        
    #     # Ensure minimum height
    #     position[2] = max(0.5, position[2])
        
    #     return position


    def initialize_drones_closer(self, current_gate=0, offset_distance=2.0, jitter_range=0.5, evaluation_type='normal'):
        """
        Initialize drones in front of the first gate, oriented toward the gate.
        
        Args:
            current_gate: Gate index to position drones near
            offset_distance: Base distance from gate
            jitter_range: Amount of random variation to apply
            evaluation_type: Type of evaluation scenario:
                - 'normal': Standard race start
                - 'tight': Very close positioning for collision avoidance testing
                - 'overtaking': Setup to test overtaking behavior
                - 'approach': Test gate approach precision
        
        Returns:
            position: 3D position for drone
        """
        # Select the gate
        gate = self.gates[current_gate]
        gate_center = gate['center']
        gate_yaw = gate['yaw']
        
        # Direction vectors - adjust the forward direction to point TOWARD the gate
        # This is the key change - using negative direction to place drones in front
        forward_dir = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
        right_dir = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
        up_dir = np.array([0.0, 0.0, 1.0])

        # Set parameters based on evaluation type
        if evaluation_type == 'normal':
            # Standard race start - reasonable distance in front of gate
            base_distance = offset_distance
            lateral_range = jitter_range  # How far drones spread laterally
            height_range = jitter_range * 0.6   # How much height variation
            
        elif evaluation_type == 'tight':
            # Very close positioning to test collision avoidance
            base_distance = offset_distance * 0.5
            lateral_range = jitter_range * 0.4  # Tighter lateral spacing
            height_range = jitter_range * 0.3   # Minimal height variation
            
        elif evaluation_type == 'overtaking':
            # Positioning that encourages overtaking scenarios
            base_distance = offset_distance * 1.5
            lateral_range = jitter_range * 0.6  # Moderate lateral spacing
            height_range = jitter_range * 0.4   # Some height variation
            
        elif evaluation_type == 'approach':
            # Test precise gate approach
            base_distance = offset_distance * 2.0
            lateral_range = jitter_range * 1.5  # Wider lateral spacing
            height_range = jitter_range  # More height variation to test approach angles
        else:
            # Default to normal
            base_distance = offset_distance
            lateral_range = jitter_range
            height_range = jitter_range * 0.6
        
        # Determine next gate to calculate proper orientation
        next_gate_idx = (current_gate + 1) % len(self.gates)
        next_gate_center = self.gates[next_gate_idx]['center']
        
        # Calculate vector from current gate to next gate
        gate_to_next = next_gate_center - gate_center
        gate_to_next = gate_to_next / np.linalg.norm(gate_to_next)
        
        # Compute position in front of the gate (opposite side from next gate)
        # This ensures drones are positioned to approach the gate from the correct direction
        base_position = gate_center - gate_to_next * base_distance
        
        # Add lateral offset based on drone ID or random
        # This creates a line of drones in front of the gate
        if hasattr(self, 'drone_id') and hasattr(self, 'num_drones'):
            # If we know which drone this is out of how many
            drone_id = self.drone_id
            num_drones = self.num_drones
            
            # Spread drones evenly
            if num_drones > 1:
                lateral_offset = (drone_id - (num_drones-1)/2) * lateral_range
                
                # For overtaking scenario, stagger drones forward/backward too
                if evaluation_type == 'overtaking':
                    forward_offset = (drone_id % 2) * 1.0  # Alternate drones forward/back
                    base_position = base_position - gate_to_next * forward_offset
            else:
                lateral_offset = 0
        else:
            # If we don't have drone ID, use random offset
            lateral_offset = np.random.uniform(-lateral_range, lateral_range)
        
        # Calculate proper lateral direction (perpendicular to gate-to-next vector)
        lateral_dir = np.array([-gate_to_next[1], gate_to_next[0], 0.0])
        
        # Apply lateral and height offsets
        height_offset = np.random.uniform(-height_range, height_range)
        position = (base_position + 
                lateral_dir * lateral_offset + 
                up_dir * height_offset)
        
        # Ensure minimum height
        position[2] = max(0.5, position[2])
        
        # Calculate proper orientation (looking toward the gate)
        if hasattr(self, 'initial_orientation'):
            # If the environment allows setting initial orientation
            # Calculate quaternion that looks from drone position toward gate
            direction = gate_center - position
            direction[2] = 0  # Keep level in Z axis
            direction = direction / np.linalg.norm(direction)
            
            # Calculate yaw angle
            yaw = np.arctan2(direction[1], direction[0])
            
            # Set initial orientation - if your environment uses quaternions or euler angles
            self.initial_orientation = yaw
        
        return position
    
    def transform_to_gate_coordinates(self, drone_position, gate):
        """
        Transform drone position from world coordinates to gate's local coordinate system.
        
        Args:
            drone_position: 3D position of the drone [x, y, z]
            gate: Dictionary containing gate center and yaw
            
        Returns:
            local_position: Drone position in gate's local coordinate system
        """
        # Gate information
        gate_center = gate["center"]
        gate_yaw = gate["yaw"]
        
        # Step 1: Translate - move origin to gate center
        translated = drone_position - gate_center
        
        # Step 2: Rotate around Z axis by -gate_yaw to align with gate orientation
        cos_yaw = np.cos(-gate_yaw)
        sin_yaw = np.sin(-gate_yaw)
        
        # Create rotation matrix for Z-axis rotation
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to get local coordinates
        local_position = rotation_matrix @ translated
        
        return local_position
    
    def check_gate_passing(self, drone_position, gate):
        """
        Check if a drone has passed through a gate.
        
        Args:
            drone_position: 3D position of the drone
            gate: Dictionary containing gate information
            
        Returns:
            boolean: True if drone is passing through gate, False otherwise
        """
        # Transform drone position to gate's local coordinate system
        local_position = self.transform_to_gate_coordinates(drone_position, gate)
        
        # Gate dimensions
        gate_width = self.gate_size
        gate_height = self.gate_size
        
        # Check if drone is passing through the gate plane
        passing_through_plane = abs(local_position[0]) < self.gate_passing_tolerance
        
        # Check if drone is within the gate's opening
        within_width = abs(local_position[1]) < gate_width/2
        within_height = abs(local_position[2]) < gate_height/2
        
        return passing_through_plane and within_width and within_height
    
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

            
            opponents_info = {}
            for other_agent in self.vehicles:
                if other_agent != agent:
                    opponents_info[other_agent] = {
                        "position": self.vehicles[other_agent]["drone"].state[0:3],
                    }

            # Compute reward based on the chosen reward function.
            if self.reward_type == 3:
                reward = self.reward_function_3(distance, vehicle["prev_distance"], velocity, unit_rel_pos, opponents_info, position)
            else:
                reward = self.reward_function(distance, vehicle["prev_distance"], velocity, unit_rel_pos)

            # Check if the target gate is reached.
            # dx = position[0] - current_gate_position[0]
            # dy = position[1] - current_gate_position[1]
            # dz = position[2] - current_gate_position[2]

            # cos_yaw = np.cos(-current_gate_yaw)
            # sin_yaw = np.sin(-current_gate_yaw)
            # rx = cos_yaw * dx - sin_yaw * dy
            # ry = sin_yaw * dx + cos_yaw * dy
            # # Here, we assume that the gate is upright, so the vertical (z) axis is already aligned.
            # rz = dz
            # in_xy = (abs(rx) <= self.gate_passing_tolerance) and (abs(ry) <= self.gate_passing_tolerance)
            # z_diff = abs(position[2] - current_gate_position[2])
            # in_z = (z_diff <= self.gate_passing_tolerance)
            # target_reached = in_xy and in_z

            current_gate = self.gates[vehicle["current_target_index"]]
            target_reached = self.check_gate_passing(position, current_gate)

            if target_reached:
                reward = 1.0
                # Move to the next gate (cyclic order) and increment progress.
                vehicle["current_target_index"] = (vehicle["current_target_index"] + 1) % self.num_targets
                vehicle["progress"] += 1
                target_reached = False

                # Get the new target position after changing the target index
                new_target_idx = vehicle["current_target_index"]
                new_target_position = self.gate_positions[new_target_idx]
                
                # Calculate and update the initial distance to the new target
                new_distance = np.linalg.norm(new_target_position - position)
                vehicle["prev_distance"] = new_distance  # Reset prev_distance for the new target
            
            else:
                # Save the current distance for reward shaping on the next step.
                vehicle["prev_distance"] = distance



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

            
            
            vehicle["reward"] = reward

            rewards[agent] = reward
            dones[agent] = done


        # --- Collision Penalty (Curriculum Controlled) ---
        if self.enable_collision:
            # Get positions of all drones
            positions = {
                drone_id: self.vehicles[drone_id]["drone"].state[:3]
                for drone_id in self.vehicles
            }

            # Check collisions between all pairs of drones
            for i, (drone_i, pos_i) in enumerate(positions.items()):
                for drone_j, pos_j in list(positions.items())[i+1:]:
                    if np.linalg.norm(pos_i - pos_j) < self.drone_collision_margin:
                        # Apply collision penalty to both drones involved
                        rewards[drone_i] -= self.collision_penalty
                        rewards[drone_j] -= self.collision_penalty
                        infos[drone_i]["collision"] = 1
                        infos[drone_j]["collision"] = 1
                        if self.terminate_on_collision and infos[self.training_agent]["collision"]:
                            dones["__all__"] = True

        # --- Overtaking Bonus (Curriculum Controlled) ---
        if self.enable_overtake:
            # For each drone, check if it has passed another drone since the last step
            for agent in self.agents:
                current_pos = self.vehicles[agent]["drone"].state[:3]
                current_vel_dir = self.vehicles[agent]["drone"].state[3:6]
                if np.linalg.norm(current_vel_dir) > 0:
                    current_vel_dir = current_vel_dir / np.linalg.norm(current_vel_dir)
                
                # Check against all other drones
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_pos = self.vehicles[other_agent]["drone"].state[:3]
                        
                        # Vector from agent to other drone
                        rel_vec = other_pos - current_pos
                        
                        # Project this vector onto agent's velocity direction
                        projection = np.dot(rel_vec, current_vel_dir)
                        
                        # If this projection was negative last step but is positive now,
                        # it means the agent has passed the other drone
                        key = f"rel_projection_{agent}_{other_agent}"
                        if key in self.prev_state_info:
                            prev_projection = self.prev_state_info[key]
                            if prev_projection < 0 and projection > 0:
                                rewards[agent] += self.overtake_reward
                        
                        # Store current projection for next step
                        if not hasattr(self, 'prev_state_info'):
                            self.prev_state_info = {}
                        self.prev_state_info[key] = projection


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
    
    def reward_function_3(self, distance, prev_distance, velocity, unit_rel_pos, opponents_info, agent_position):
        # Original reward components
        x = distance / self.max_distance
        base_reward = 2 * (np.exp(-self.a * x) - np.exp(-self.a)) / (1 - np.exp(-self.a)) - 1
        delta_distance = prev_distance - distance
        bonus_distance = 2.5 * delta_distance
        vel_norm = np.linalg.norm(velocity)

        # Speed reward/penalty
        desired_speed = self.minimum_velocity
        if vel_norm <= desired_speed:
            speed_gap = vel_norm - self.minimum_velocity
        else:
            speed_gap = desired_speed - vel_norm

        # Direction deviation penalty
        if vel_norm > 1e-6:
            cos_angle = np.dot(velocity, unit_rel_pos) / (vel_norm + 1e-6)
            deviation = 1.0 - cos_angle
        else:
            deviation = 0.0

        # New collision avoidance term
        collision_penalty = 0
        safety_radius = self.drone_collision_margin * 3 # Define minimum safe distance between drones

        for drone_id, info in opponents_info.items():
            opponent_pos = info["position"]
            distance_to_opponent = np.linalg.norm(agent_position - opponent_pos)
            
            if distance_to_opponent < safety_radius:
                # Exponential penalty that increases as drones get closer
                collision_penalty += np.exp(safety_radius - distance_to_opponent) - 1

        collision_penalty = collision_penalty if collision_penalty > 0 else 0
        # Combine all reward components
        reward = (self.w_distance * base_reward +
                self.w_distance_change * bonus_distance -
                self.w_deviation * deviation +
                self.w_inactivity * speed_gap -
                self.w_collision_penalty * collision_penalty)  # Add collision penalty

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

        # print("Shapes of arrays:")
        # print("normalized_relative_position:", np.array(normalized_relative_position).shape)
        # print("normalized_distance:", np.array(normalized_distance).shape)
        # print("normalized_position:", np.array(normalized_position).shape)
        # print("normalized_velocity:", np.array(normalized_velocity).shape)
        # print("projected_vel:", np.array(projected_vel).shape)
        # print("normalized_speed:", np.array(normalized_speed).shape)
        # print("normalized_delta_velocity:", np.array(normalized_delta_velocity).shape)
        # print("normalized_angular_speed:", np.array(normalized_angular_speed).shape)
        # print("normalized_angular_velocity:", np.array(normalized_angular_velocity).shape)
        # print("one_hot:", np.array(one_hot).shape)
        # print("sin_yaw:", np.array(sin_yaw).shape)
        # print("cos_yaw:", np.array(cos_yaw).shape)
        # print("gate_ratios:", np.array(gate_ratios).shape)
        # print("normalized_time:", np.array(normalized_time).shape)
        # print("normalized_heading_error:", np.array(normalized_heading_error).shape)
        # print("normalized_progress:", np.array(normalized_progress).shape)
        # print("all_opponent_rel_pos:", np.array(all_opponent_rel_pos).shape)
        # print("all_opponent_distances:", np.array(all_opponent_distances).shape)
        # print("all_opponent_velocities:", np.array(all_opponent_velocities).shape)
        # print("all_opponent_progresses:", np.array(all_opponent_progresses).shape)

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



        # obs = np.concatenate([
        #     normalized_relative_position,      # 3 values.
        #     normalized_distance,               # 1 value.
        #     normalized_position,               # 3 values.
        #     normalized_velocity,               # 3 values.
        #     projected_vel,                     # 1 value.
        #     normalized_speed,                  # 1 value.
        #     normalized_delta_velocity,         # 3 values.
        #     normalized_angular_speed,          # 1 value.
        #     normalized_angular_velocity,       # 3 values.
        #     one_hot,                           # num_targets values.
        #     sin_yaw,                           # 1 value.
        #     cos_yaw,                           # 1 value.
        #     gate_ratio,                        # 1 value.
        #     normalized_time,                   # 1 value.
        #     normalized_heading_error,          # 1 value.
        #     normalized_progress,               # 1 value.
        #     normalized_opponent_rel_pos,       # 3 values.
        #     normalized_opponent_distance,      # 1 value.
        #     normalized_opponent_velocity,      # 3 values.
        #     normalized_opponent_progress       # 1 value.
        # ])

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

    def render(self, mode='human'):
        for agent in self.agents:
            reward = self.vehicles[agent]["reward"]
            pos = self.vehicles[agent]["drone"].state[:3]
            velocity = self.vehicles[agent]["drone"].state[3:6]
            current_target = self.gate_positions[self.vehicles[agent]["current_target_index"]]
            distance = np.linalg.norm(current_target - pos)
            print(f"Agent: {agent} | Dist: {distance:.2f}  | reward: {reward:.2f} | Target Index: {self.vehicles[agent]['current_target_index']} | "
                  f"Target: {current_target} | Step: {self.current_step} | Position: {pos} | Velocity: {velocity}")

    def get_state(self, agent):
        return self.vehicles[agent]["drone"].state




