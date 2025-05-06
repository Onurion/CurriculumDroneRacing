import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dynamics.drone_dynamics import DroneDynamics, LowLevelController, VelocityTracker
import numpy as np
from collections import deque


class DroneRaceCurriculumMultiEnv_v10(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_agents: int=2,
                 n_gates: int = 5,
                 radius: float = 10.0,
                 max_steps: int = 2000,
                 dt: float = 0.1,
                 gate_size: float = 1.0,
                 evaluation_mode: bool=False,
                 random_init: bool=True,
                 track_type: str="circle",
                 # Speed parameters
                 minimum_velocity: float=0.2,
                 action_coefficient: float=1.0,
                 # obs act parameters
                 reward_type: int=2,
                 observation_type: int=2,
                 buffer_size: int=10,
                 is_buffer_obs: bool=False,
                 # collision parameters:
                 enable_collision: bool=False,
                 terminate_on_collision: bool=False,
                 collision_penalty: float = 0.5,
                 drone_collision_margin: float =0.5,
                 gate_passing_tolerance: float = 0.5,
                 safety_distance:float=  2.0,
                 proximity_penalty:float = 0.1,
                 # reward coefficients
                 enable_overtake:bool=False,
                 waypoint_radius:float = 0.8,
                 waypoint_reward:float = 0.2,
                 catchup_bonus:float = 0.3,
                 distance_reward_factor:float = 2.5,
                 alignment_reward_factor:float = 0.3,
                 overtake_reward: float =0.0,
                 # reward scaling
                 tanh_scaling_factor:float = 0.4,
                 jitter_range:list = [0.25, 0.75]):
        super(DroneRaceCurriculumMultiEnv_v10, self).__init__()

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
        self.eval_scenarios = ["overtaking", "tight", "overtaking", "approach", "normal"]
        # self.eval_scenarios = ["normal"]

        # Curriculum parameters.
        self.action_coefficient = action_coefficient          # Multiplier applied to the raw action.
        self.minimum_velocity = minimum_velocity     # Minimum velocity (used to penalize inactivity).
        self.enable_collision = enable_collision  # Whether collision detection is enabled.
        self.collision_penalty = collision_penalty  # Penalty for collision.
        self.drone_collision_margin = drone_collision_margin  # Drone Collision threshold in meters.
        self.terminate_on_collision = terminate_on_collision  # Terminate episode on collision.
        self.enable_overtake = enable_overtake    # Whether overtake bonus is enabled.
        self.overtake_reward = overtake_reward    # Reward bonus for overtaking in selfplay.
        self.tanh_scaling_factor = tanh_scaling_factor
        

        # Initialize with reasonable parameter values if not already set
        self.transition_period = 3  # Number of steps for reward transition
        self.waypoint_radius = waypoint_radius  # Radius around waypoints to detect passing
        self.waypoint_reward = waypoint_reward  # Reward for passing through waypoints
        self.safety_distance = safety_distance  # Distance at which to begin proximity warnings
        self.proximity_penalty = proximity_penalty  # Maximum penalty for proximity violations
        self.catchup_bonus = catchup_bonus  # Bonus for trailing drones when passing gates
        self.distance_reward_factor = distance_reward_factor  # Reward multiplier for distance improvement
        self.alignment_reward_factor = alignment_reward_factor  # Reward factor for velocity alignment
        self.gate_reward = 1.0  # Reward for passing through a gate
        self.out_of_bounds_penalty = 1.0
        self.attitude_failure_penalty = 1.0
        self.max_reward = 1.0
        
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
        # Generate racing line waypoints if not provided
        self.generate_racing_line_waypoints()

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
        
        # Updated gate positions
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
            initial_distance = distances[vehicle["current_target_index"]]
            vehicle["distance_to_gate"] = initial_distance
            # Add a small buffer to prevent initial spike
            vehicle["prev_distance"] = initial_distance + 0.01  # just slightly higher
            # Option 2: Add a "first_step" flag to skip distance reward on first step
            vehicle["first_step"] = True

            vehicle["prev_velocity"] = np.zeros(3, dtype=np.float32)
            vehicle["progress"] = 0
            vehicle["prev_progress"] = 0
            vehicle["reward"] = 0
            self.vehicles[agent]["collisions"] = 0

            
            # Reset distance tracking
            initial_position = self.vehicles[agent]["drone"].state[:3]
            initial_target = vehicle["current_target_index"]
            self.vehicles[agent]["prev_distance"] = np.linalg.norm(initial_target - initial_position)
            
            # Reset transition tracking
            self.vehicles[agent]["transition_steps"] = 0
            self.vehicles[agent]["last_gate_distance"] = 0
            
            # Reset waypoint flags
            if hasattr(self, 'waypoints'):
                for target_idx in range(self.num_targets):
                    for wp_idx in range(len(self.waypoints[target_idx])):
                        self.vehicles[agent][f"passed_waypoint_{target_idx}_{wp_idx}"] = False

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
    

    def calculate_reward(self, drone_id, position, velocity, target_reached=False):
        """Calculate reward for a drone based on its state and progress."""
        vehicle = self.vehicles[drone_id]
        target_idx = vehicle["current_target_index"]
        target_position = self.gate_positions[target_idx]
        reward_components = {}

        reward_components = {"gate": 0, "waypoint": 0, "alignment":0 ,
                            "distance":0, "proximity": 0, "catchup": 0}
        
        # Calculate distance to current target
        distance = np.linalg.norm(target_position - position)
        
        # Update reward based on distance change from previous step
        prev_distance = vehicle["prev_distance"]
        delta_distance = prev_distance - distance

        
        
        # Smooth reward transition after passing a gate
        if vehicle["transition_steps"] > 0:
            # Calculate transition weighting (starts at 1.0 and decreases to 0)
            transition_weight = vehicle["transition_steps"] / self.transition_period
            
            # Calculate distance and delta for previous target
            prev_target_idx = (target_idx - 1) % self.num_targets
            prev_target_position = self.gate_positions[prev_target_idx]
            prev_target_distance = np.linalg.norm(prev_target_position - position)
            
            # Calculate how much we've moved away from the previous target
            # Note: For the previous target, moving away is good (positive value)
            prev_delta = prev_target_distance - vehicle["last_gate_distance"]
            
            # Blend the rewards: weight * prev_target + (1-weight) * current_target
            # As transition_weight decreases, we shift from rewarding movement away from
            # previous gate to rewarding movement toward the current gate
            blended_delta = (transition_weight * prev_delta + 
                            (1 - transition_weight) * delta_distance)
            
            # Use the blended delta instead
            delta_distance = blended_delta
            
            # Decrement transition counter
            vehicle["transition_steps"] -= 1
        
        # If using Option 2, modify your distance reward code:
        if vehicle.get("first_step", False):
            distance_reward = 0
            vehicle["first_step"] = False
        else:
            # Basic progress reward (moving toward target)
            distance_reward = self.distance_reward_factor * delta_distance

        reward_components["distance"] = distance_reward
        reward = distance_reward
        
        if target_reached:
            # Gate passing reward
            reward += self.gate_reward
            # Set up transition period
            vehicle["transition_steps"] = self.transition_period
            vehicle["last_gate_distance"] = distance
            reward_components["gate"] = self.gate_reward
        
        # --- Racing Line Guidance ---
        # Check if the drone is passing through any waypoints
        if hasattr(self, 'waypoints'):
            for waypoint_idx, waypoint in enumerate(self.waypoints[target_idx]):
                waypoint_key = f"passed_waypoint_{target_idx}_{waypoint_idx}"
                waypoint_dist = np.linalg.norm(waypoint - position)
                
                # If drone is close to waypoint and hasn't passed it yet
                if waypoint_dist < self.waypoint_radius and not vehicle.get(waypoint_key, False):
                    reward += self.waypoint_reward
                    vehicle[waypoint_key] = True
                    reward_components["waypoint"] = self.waypoint_reward
        
        # --- Velocity Direction Component ---
        # Reward alignment between velocity and direction to target
        if np.linalg.norm(velocity) > 0.1:  # Only if drone is moving
            direction_to_target = target_position - position
            if np.linalg.norm(direction_to_target) > 0:
                direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
                velocity_norm = velocity / np.linalg.norm(velocity)
                alignment = np.dot(velocity_norm, direction_to_target)
                # Reward ranges from -0.5 to 0.5 based on alignment
                alignment_reward = self.alignment_reward_factor * alignment
                reward += alignment_reward
                reward_components["alignment"] = alignment_reward
        
        # --- Proximity-Based Collision Avoidance ---
        # Add penalties for being too close to other drones
        for other_id in self.vehicles:
            if other_id != drone_id:
                other_position = self.vehicles[other_id]["drone"].state[:3]
                distance_to_other = np.linalg.norm(position - other_position)
                
                # Apply graduated penalty based on proximity
                if distance_to_other < self.drone_collision_margin:
                    # Already collided - severe penalty
                    collision_penalty = self.collision_penalty
                elif distance_to_other < self.safety_distance:
                    # In danger zone - graduated penalty (stronger as distance decreases)
                    danger_ratio = (self.safety_distance - distance_to_other) / (self.safety_distance - self.drone_collision_margin)
                    collision_penalty = self.proximity_penalty * danger_ratio**2  # Quadratic scaling
                else:
                    collision_penalty = 0
                    
                reward -= collision_penalty
                reward_components["proximity"] = - collision_penalty
    

        # --- Catch-Up Mechanics ---
        # Apply small boost to trailing drones when they pass gates
        if target_reached and not self.is_leading_drone(drone_id):
            # Already gave +1.0 for passing gate, add catch-up bonus
            reward += self.catchup_bonus
            reward_components["catchup"] = self.catchup_bonus
        
        # Store current distance for next step
        vehicle["prev_distance"] = distance

        scaled_reward = np.tanh(reward * self.tanh_scaling_factor)

        
        
        # Clip final reward to desired range
        return scaled_reward, reward_components

    

    def is_leading_drone(self, drone_id):
        """Determine if this drone is in the lead."""
        # Get progress of current drone
        current_progress = self.vehicles[drone_id]["progress"]
        current_target = self.vehicles[drone_id]["current_target_index"]
        
        # For same progress, compare distance to next gate (closer is better)
        current_position = self.vehicles[drone_id]["drone"].state[:3]
        current_target_pos = self.gate_positions[current_target]
        current_distance = np.linalg.norm(current_target_pos - current_position)
        
        # Check if any other drone has made more progress
        for other_id in self.vehicles:
            if other_id != drone_id:
                other_progress = self.vehicles[other_id]["progress"]
                
                # If other drone has more gate completions, current drone isn't leading
                if other_progress > current_progress:
                    return False
                
                # If tied on progress, check distance to next gate
                elif other_progress == current_progress:
                    other_target = self.vehicles[other_id]["current_target_index"]
                    other_position = self.vehicles[other_id]["drone"].state[:3]
                    other_target_pos = self.gate_positions[other_target]
                    other_distance = np.linalg.norm(other_target_pos - other_position)
                    
                    # If other drone is closer to its next gate, current drone isn't leading
                    if other_distance < current_distance:
                        return False
        
        # This drone is in the lead
        return True

    def generate_racing_line_waypoints(self):
        """Generate waypoints forming racing lines between gates."""
        self.waypoints = {}
        
        for i in range(self.num_targets):
            current_gate = self.gate_positions[i]
            next_gate = self.gate_positions[(i + 1) % self.num_targets]
            
            # Calculate vector between gates
            gate_vector = next_gate - current_gate
            gate_distance = np.linalg.norm(gate_vector)
            
            # Create 2 waypoints for each gate transition
            waypoints = []
            
            # First waypoint: 1/3 of the way between gates, slightly offset to create a racing line
            first_wp = current_gate + gate_vector * 0.33
            # Add a lateral offset to create a smoother racing line
            perpendicular = np.array([-gate_vector[1], gate_vector[0], 0])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular) * (gate_distance * 0.1)
                first_wp += perpendicular
            
            # Second waypoint: 2/3 of the way between gates, with opposite offset
            second_wp = current_gate + gate_vector * 0.66
            if np.linalg.norm(perpendicular) > 0:
                second_wp -= perpendicular
            
            waypoints.append(first_wp)
            waypoints.append(second_wp)
            
            self.waypoints[i] = waypoints

    def step(self, actions):
        # Initialize data structures once
        rewards = {agent: 0.0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {
            "collision": 0,
            "total_collisions": self.vehicles[agent].get("collisions", 0)
        } for agent in self.agents}
        
        # Increment the global step count
        self.current_step += 1
        
        # --- SINGLE MAIN LOOP: Update all drone positions & collect state data ---
        # Store position data for collision detection
        positions = {}
        velocities = {}
        directions = {}
        
        for agent in self.agents:
            # Process action and update drone state
            act = np.clip(actions[agent], self.action_space[agent].low, self.action_space[agent].high)
            act = self.action_coefficient * act
            
            vehicle = self.vehicles[agent]
            drone = vehicle["drone"]
            velocity = drone.state[3:6]
            desired_velocity = velocity + act * self.dt
            
            control = vehicle["vel_tracker"].compute_control(np.array(desired_velocity))
            new_state, _ = drone.step(control)
            
            # Extract state variables
            position = new_state[:3]
            velocity = new_state[3:6]
            attitude = new_state[6:9]  # roll, pitch, yaw
            
            # Store for later use
            positions[agent] = position
            velocities[agent] = velocity
            vehicle["position"] = position
            vehicle["velocity"] = velocity
            vehicle["attitude"] = attitude
            
            # Normalize velocity direction for later use
            vel_norm = np.linalg.norm(velocity)
            if vel_norm > 0.1:
                directions[agent] = velocity / vel_norm
            
            # Check target gate
            current_target_idx = vehicle["current_target_index"]
            current_gate = self.gates[current_target_idx]
            current_gate_position = self.gate_positions[current_target_idx]
            
            # Calculate distance to target
            distance = np.linalg.norm(current_gate_position - position)
            target_reached = self.check_gate_passing(position, current_gate)
            
            # Calculate reward
            if self.reward_type == 5:
                reward, reward_components = self.calculate_reward(agent, position, velocity, target_reached)
                rewards[agent] += reward
            else:
                raise ValueError("Unknown reward type")
            
            # Process gate passing
            if target_reached:
                next_target_idx = (current_target_idx + 1) % self.num_targets
                vehicle["current_target_index"] = next_target_idx
                vehicle["progress"] += 1
                
                # Reset waypoint flags efficiently
                if hasattr(self, 'waypoints'):
                    for wp_idx in range(len(self.waypoints[next_target_idx])):
                        vehicle[f"passed_waypoint_{next_target_idx}_{wp_idx}"] = False
                    
                    if next_target_idx == 0:
                        for gate_idx in range(self.num_targets):
                            for wp_idx in range(len(self.waypoints[gate_idx])):
                                vehicle[f"passed_waypoint_{gate_idx}_{wp_idx}"] = False
                
                # Update distance for new target
                new_target_position = self.gate_positions[next_target_idx]
                vehicle["prev_distance"] = np.linalg.norm(new_target_position - position)
            else:
                vehicle["prev_distance"] = distance
            
            # Check out-of-bounds
            all_distances = np.linalg.norm(self.gate_positions - position, axis=1)
            if np.min(all_distances) > self.max_distance:
                dones[agent] = True
                rewards[agent] -= self.out_of_bounds_penalty
                reward_components["out_of_bounds"] = -self.out_of_bounds_penalty
            
            # Check attitude failure
            roll, pitch = attitude[0], attitude[1]
            if abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
                vehicle["attitude_failures"] = vehicle.get("attitude_failures", 0) + 1
                rewards[agent] -= self.attitude_failure_penalty
                reward_components["attitude_failure"] = -self.attitude_failure_penalty
                infos[agent]["attitude_failure"] = 1
                dones[agent] = True
            
            # Update info dictionary
            infos[agent].update({
                "progress": vehicle["progress"],
                "velocity": vel_norm,
                "lap": vehicle["progress"] // self.num_targets + 1,
                "attitude_failures": vehicle.get("attitude_failures", 0),
                "overtakes": vehicle.get("overtakes", 0),
                "reward_components": reward_components
            })
        
        # --- Collision Detection (Single N²/2 loop) ---
        if hasattr(self, 'enable_collision') and self.enable_collision:
            agents_list = list(self.agents)
            for i in range(len(agents_list)):
                agent = agents_list[i]
                agent_pos = positions[agent]
                
                for j in range(i+1, len(agents_list)):
                    other_agent = agents_list[j]
                    other_pos = positions[other_agent]
                    
                    if np.linalg.norm(agent_pos - other_pos) < self.drone_collision_margin:
                        # Update collision counters
                        self.vehicles[agent]["collisions"] = self.vehicles[agent].get("collisions", 0) + 1
                        self.vehicles[other_agent]["collisions"] = self.vehicles[other_agent].get("collisions", 0) + 1                        
                        
                        # Update info
                        infos[agent]["collision"] = 1
                        infos[other_agent]["collision"] = 1
                        infos[agent]["total_collisions"] = self.vehicles[agent]["collisions"]
                        infos[other_agent]["total_collisions"] = self.vehicles[other_agent]["collisions"]

                        # Terminate if configured
                        if self.terminate_on_collision:
                            dones[agent] = True
                            dones[other_agent] = True
        
        # --- Overtaking Detection (Optimized) ---
        if hasattr(self, 'enable_overtake') and self.enable_overtake:
            # Create the previous projections dictionary if it doesn't exist
            if not hasattr(self, 'prev_state_info'):
                self.prev_state_info = {}
            
            # Calculate current projections and check for overtakes
            current_projections = {}
            agents_list = list(self.agents)
            
            for i in range(len(agents_list)):
                agent = agents_list[i]
                if agent not in directions:  # Skip if velocity too low
                    continue
                    
                agent_pos = positions[agent]
                agent_dir = directions[agent]
                
                for j in range(len(agents_list)):
                    if i == j:
                        continue
                        
                    other_agent = agents_list[j]
                    other_pos = positions[other_agent]
                    
                    # Calculate projection
                    rel_vec = other_pos - agent_pos
                    projection = np.dot(rel_vec, agent_dir)
                    
                    # Store for next step
                    key = f"rel_projection_{agent}_{other_agent}"
                    current_projections[key] = projection
                    
                    # Check for overtake
                    if key in self.prev_state_info:
                        prev_projection = self.prev_state_info[key]
                        if prev_projection < 0 and projection > 0:
                            rewards[agent] += self.overtake_reward
                            self.vehicles[agent]["overtakes"] = self.vehicles[agent].get("overtakes", 0) + 1
                            infos[agent]["overtakes"] = self.vehicles[agent]["overtakes"]
                            infos[agent]["reward_components"]["overtake"] = self.overtake_reward
            
            # Update projections for next step
            self.prev_state_info.update(current_projections)
        
        # Clip rewards and update progress
        for agent in self.agents:
            final_reward = np.clip(rewards[agent], -self.max_reward, self.max_reward)
            self.vehicles[agent]["reward"] = final_reward
            rewards[agent] = final_reward
            self.vehicles[agent]["prev_progress"] = self.vehicles[agent]["progress"]
        
        # Handle episode termination
        global_done = self.current_step >= self.max_steps
        dones["__all__"] = global_done or any(dones.values())
        truncated = global_done
        
        # Get observations (unavoidable loop - needs to happen after all updates)
        obs = {}
        for agent in self.agents:
            current_obs = self.observation_function(agent)
            
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

    def render(self, info, mode='human'):
        for agent in self.agents:
            reward = self.vehicles[agent]["reward"]
            pos = self.vehicles[agent]["drone"].state[:3]
            velocity = self.vehicles[agent]["drone"].state[3:6]
            current_target = self.gate_positions[self.vehicles[agent]["current_target_index"]]
            distance = np.linalg.norm(current_target - pos)
            print(f"Agent: {agent} | Dist: {distance:.2f}  | reward: {reward:.2f} | Target Index: {self.vehicles[agent]['current_target_index']} | "
                  f"Target: {current_target} | Step: {self.current_step} | Position: {pos} | Velocity: {velocity}")
            
            reward_components = info[agent]["reward_components"]
            print (f"Reward components of {agent}:")
            txt = ""
            for k in reward_components.keys():
                txt += f"{k}: {reward_components[k]:.2f} "

            print (txt)

    def get_state(self, agent):
        return self.vehicles[agent]["drone"].state




