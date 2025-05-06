import numpy as np
from scipy.interpolate import CubicSpline

class RacingLineManager:
    """
    Manages generation and selection of racing lines for drone racing environments.
    
    This class handles the creation of multiple racing line strategies, adaptive line selection,
    and provides utilities for racing line management and visualization.
    """
    
    def __init__(self, environment, config=None):
        """
        Initialize the racing line manager.
        
        Args:
            environment: Reference to the main environment (for accessing gates, etc.)
            config: Dictionary of configuration parameters
        """
        self.env = environment
        
        # Default configuration
        self.config = {
            'enable_multiple_lines': True,
            'enable_adaptive_selection': True,
            'visualize_lines': False,
            'line_types': ['direct', 'standard', 'overtake', 'recovery'],
            'points_per_gate': 5,  # Number of points to generate between gates
            'gate_radius': 1.0,    # Default gate size if not specified by environment
            'transition_period': 10 # Steps for reward transition when passing gates
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Set properties based on config
        self.enable_multiple_lines = self.config['enable_multiple_lines']
        self.enable_adaptive_selection = self.config['enable_adaptive_selection']
        self.visualize_lines = self.config['visualize_lines']
        
        # Initialize storage for racing lines and visualization
        self.racing_lines = {}
        self.visualization_markers = []
        
        # Generate racing lines
        self.initialize_racing_lines()
    
    def initialize_racing_lines(self):
        """Generate all racing lines based on the track layout."""
        # Generate default waypoints if not provided
        if not hasattr(self, 'default_waypoints'):
            self.default_waypoints = self.generate_default_waypoints()
        
        # Generate different racing line variants
        if self.enable_multiple_lines:
            # Direct/aggressive racing line
            self.racing_lines["direct"] = self.generate_racing_line(
                self.default_waypoints, 
                risk_factor=0.9,  # More aggressive, closer to gates
                smoothing=0.7     # Less smoothing for more direct path
            )
            
            # Standard racing line
            self.racing_lines["standard"] = self.generate_racing_line(
                self.default_waypoints,
                risk_factor=0.7,  # Standard distance from gates
                smoothing=0.8     # Standard smoothing
            )
            
            # Overtaking racing line
            self.racing_lines["overtake"] = self.generate_racing_line(
                self.default_waypoints,
                risk_factor=0.6,  # Wider turns for overtaking
                smoothing=0.8,    # Standard smoothing
                lateral_offset=0.8 # Offset from standard line for passing
            )
            
            # Recovery racing line
            self.racing_lines["recovery"] = self.generate_racing_line(
                self.default_waypoints,
                risk_factor=0.5,  # Safe, wider lines
                smoothing=0.9,    # Extra smoothing for stable flight
                height_bonus=0.3  # Fly slightly higher to avoid collisions
            )
        else:
            # Just use the standard line if multiple lines disabled
            self.racing_lines["standard"] = self.generate_racing_line(
                self.default_waypoints,
                risk_factor=0.7,
                smoothing=0.8
            )
        
        # Generate racing line direction vectors for alignment rewards
        self.calculate_racing_line_directions()
        
        # Update visualization if enabled
        if self.visualize_lines:
            self.update_visualization()
    
    def generate_default_waypoints(self):
        """Extract gate centers as default waypoints."""
        waypoints = []
        
        # Access gates through environment
        if hasattr(self.env, 'gates'):
            for gate in self.env.gates:
                gate_center = gate['center']
                waypoints.append(gate_center)
        elif hasattr(self.env, 'targets'):
            # Alternative access for target-based environments
            for target in self.env.targets:
                waypoints.append(target['position'])
        else:
            # Default fallback
            print("Warning: No gates or targets found for racing line generation")
            return np.array([[0, 0, 0]])
            
        return np.array(waypoints)
    
    def generate_racing_line(self, base_waypoints, risk_factor=0.7, smoothing=0.8, 
                        lateral_offset=0.0, height_bonus=0.0):
        """
        Generate a racing line based on the base waypoints.
        
        Args:
            base_waypoints: List of 3D waypoint positions
            risk_factor: How aggressive/close to gates (0.0-1.0)
            smoothing: How much to smooth the path (0.0-1.0)
            lateral_offset: Lateral offset for alternative lines
            height_bonus: Extra height for this line
            
        Returns:
            List of 3D waypoint positions defining the racing line
        """
        if len(base_waypoints) < 2:
            return base_waypoints.copy()
        
        # Create a smoothed path through gates
        racing_line = []
        
        # First, optimize gate approach angles
        optimized_points = self.optimize_gate_approaches(
            base_waypoints, risk_factor, lateral_offset
        )
        
        # Apply smoothing if we have enough points
        if len(optimized_points) >= 4:
            # FIXED: Make sure first and last points are exactly the same for periodic BC
            # Instead of stacking arrays, create a new array with the exact same first/last element
            num_points = len(optimized_points)
            extended_points = np.zeros((num_points + 2, optimized_points.shape[1]))
            extended_points[1:-1] = optimized_points
            extended_points[0] = optimized_points[-1]  # First = last original point
            extended_points[-1] = optimized_points[0]  # Last = first original point
            
            # Generate smoother path through points
            t = np.arange(len(extended_points))
            
            try:
                # Try with periodic boundary
                cs = CubicSpline(t, extended_points, bc_type='periodic')
            except ValueError:
                # Fallback to not-a-knot boundary if periodic fails
                print("Warning: Periodic spline failed, using 'not-a-knot' boundary condition")
                cs = CubicSpline(t, extended_points, bc_type='not-a-knot')
            
            # Sample points along the spline - using endpoint=False
            num_samples = max(int(len(optimized_points) / smoothing), len(optimized_points))
            # Sample from index 1 to len-1 (skipping the duplicated endpoints)
            t_fine = np.linspace(1, len(optimized_points), num_samples, endpoint=False)
            racing_line = cs(t_fine)
            
            # Add height bonus if specified
            if height_bonus > 0:
                racing_line[:, 2] += height_bonus
        else:
            # Not enough points for spline, use optimized points
            racing_line = optimized_points
            if height_bonus > 0:
                racing_line[:, 2] += height_bonus
        
        return racing_line
    
    def optimize_gate_approaches(self, waypoints, risk_factor, lateral_offset):
        """
        Optimize waypoints for better gate approaches with adaptive distances.
        
        Args:
            waypoints: Base waypoints (usually gate centers)
            risk_factor: Controls how close to gates (0.0-1.0)
            lateral_offset: Offset for alternative racing lines
            
        Returns:
            Optimized waypoint list
        """
        if len(waypoints) < 2:
            return np.array(waypoints)
        
        # Convert to numpy array if not already
        waypoints = np.array(waypoints)
        
        # Calculate approach vectors and turn angles for each gate
        directions = []
        turn_angles = []
        
        # Add direction vectors between consecutive gates
        for i in range(len(waypoints)):
            prev_idx = (i - 1) % len(waypoints)
            next_idx = (i + 1) % len(waypoints)
            
            # Calculate vectors to previous and next gate
            to_prev = waypoints[prev_idx] - waypoints[i]
            to_next = waypoints[next_idx] - waypoints[i]
            
            # Normalize
            prev_norm = np.linalg.norm(to_prev)
            next_norm = np.linalg.norm(to_next)
            
            if prev_norm > 0:
                to_prev = to_prev / prev_norm
            if next_norm > 0:
                to_next = to_next / next_norm
            
            # Calculate turn angle (radians)
            dot_product = np.dot(to_prev, to_next)
            # Clamp to [-1, 1] to avoid numerical errors
            dot_product = max(-1.0, min(1.0, dot_product))
            turn_angle = np.arccos(dot_product)
            turn_angles.append(turn_angle)
            
            # Calculate bisector direction (optimal approach direction)
            bisector = -(to_prev + to_next)
            if np.linalg.norm(bisector) > 0:
                bisector = bisector / np.linalg.norm(bisector)
            else:
                # Fallback if vectors cancel out
                bisector = np.array([1, 0, 0])
            
            directions.append(bisector)
        
        # Calculate optimized approach points
        optimized_points = []
        
        # Define reference up vector
        up_vector = np.array([0, 0, 1])
        
        # Parameters for adaptive approach distance
        angle_factor = 2.0  # Higher = more distance for sharp turns
        base_distance = self.config['gate_radius']
        if hasattr(self.env, 'gate_size'):
            base_distance = self.env.gate_size
        
        # Average drone speed if available (for dynamic adaptation)
        avg_speed = 5.0  # default assumption
        if hasattr(self.env, 'vehicles') and self.env.vehicles:
            speeds = []
            for vehicle in self.env.vehicles.values():
                if 'drone' in vehicle and hasattr(vehicle['drone'], 'state'):
                    # Assuming velocity is in indices 3,4,5
                    vel = vehicle['drone'].state[3:6] if len(vehicle['drone'].state) > 5 else None
                    if vel is not None:
                        speeds.append(np.linalg.norm(vel))
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
        
        speed_factor = 0.1  # Influence of speed on approach distance
        
        for i in range(len(waypoints)):
            gate_pos = waypoints[i]
            approach_dir = directions[i]
            turn_angle = turn_angles[i]
            
            # Calculate perpendicular vector using cross product
            perp_dir = np.cross(approach_dir, up_vector)
            if np.linalg.norm(perp_dir) > 1e-6:
                perp_dir = perp_dir / np.linalg.norm(perp_dir)
            else:
                # Fallback for vertical approaches
                perp_dir = np.cross(approach_dir, np.array([1, 0, 0]))
                if np.linalg.norm(perp_dir) > 1e-6:
                    perp_dir = perp_dir / np.linalg.norm(perp_dir)
                else:
                    perp_dir = np.array([0, 1, 0])
            
            # Calculate adaptive approach distance
            # More distance for sharper turns and higher speeds
            approach_distance = base_distance * (1.0 - risk_factor) * (
                1.0 + speed_factor * avg_speed + angle_factor * (turn_angle / np.pi)
            )
            
            # Apply lateral offset for alternative lines
            lateral_vector = perp_dir * lateral_offset
            
            # Calculate optimized point
            opt_point = gate_pos + approach_dir * approach_distance + lateral_vector
            
            optimized_points.append(opt_point)
        
        return np.array(optimized_points)
    
    def calculate_racing_line_directions(self):
        """Calculate and store racing line direction vectors for each segment."""
        self.racing_line_directions = {}
        
        for line_name, line in self.racing_lines.items():
            # Skip if line doesn't have enough points
            if len(line) < 2:
                continue
                
            directions = []
            for i in range(len(line)):
                # Calculate direction to next point
                next_idx = (i + 1) % len(line)
                direction = line[next_idx] - line[i]
                
                # Normalize
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    # Default direction if points are identical
                    direction = np.array([1, 0, 0])
                    
                directions.append(direction)
                
            self.racing_line_directions[line_name] = np.array(directions)
    
    def get_optimal_racing_line_point(self, drone_id, target_idx):
        """
        Get the optimal racing line point for the target index.
        
        Args:
            drone_id: ID of the drone
            target_idx: Index of the current target
            
        Returns:
            3D position for the optimal racing line point
        """
        # If multiple racing lines disabled, return default waypoint
        if not self.enable_multiple_lines:
            line_key = "standard"
        else:
            # Determine which racing line to use based on drone's position in race
            line_key = self.select_racing_line_for_drone(drone_id)
        
        # Access the selected racing line
        if line_key not in self.racing_lines:
            line_key = list(self.racing_lines.keys())[0]  # Fallback
            
        racing_line = self.racing_lines[line_key]
        
        # Map target index to racing line index
        if isinstance(racing_line, list) or isinstance(racing_line, np.ndarray):
            # For spline-generated racing lines with many points
            if len(racing_line) > len(self.default_waypoints):
                # Map the current gate index to the corresponding section of racing line
                points_per_gate = max(1, len(racing_line) // len(self.default_waypoints))
                start_idx = target_idx * points_per_gate
                
                # Use the middle point of the section for best approach
                idx = min(start_idx + points_per_gate // 2, len(racing_line) - 1)
                return racing_line[idx]
            else:
                # Direct mapping for simpler racing lines
                idx = min(target_idx, len(racing_line) - 1)
                return racing_line[idx]
        else:
            # Fallback to default waypoint
            return self.default_waypoints[target_idx]
    
    def select_racing_line_for_drone(self, drone_id):
        """
        Select the appropriate racing line based on drone's race position.
        
        Args:
            drone_id: ID of the drone
            
        Returns:
            Key for the selected racing line
        """
        # Static selection if adaptive is disabled
        if not self.enable_adaptive_selection:
            try:
                drone_num = int(drone_id.split('_')[-1])
            except (ValueError, IndexError):
                drone_num = hash(drone_id) % 100  # Fallback hash-based ID
                
            line_keys = list(self.racing_lines.keys())
            return line_keys[drone_num % len(line_keys)]
        
        # Get race ranking information
        ranks = self.get_race_ranks()
        if not ranks or drone_id not in ranks:
            return "standard"  # Default if no ranking info
            
        drone_rank = ranks.index(drone_id)
        
        # Calculate progress gap for last-place strategy
        progress_gap = 0
        if len(ranks) > 1 and drone_rank == len(ranks) - 1:
            leader_progress = self.get_drone_progress(ranks[0])
            drone_progress = self.get_drone_progress(drone_id)
            progress_gap = leader_progress - drone_progress
        
        # Select racing line based on rank and gap
        if drone_rank == 0:  # Leader
            return "direct" if "direct" in self.racing_lines else "standard"
        elif drone_rank == len(ranks)-1 and progress_gap > 1.0:  
            # Last place AND significantly behind - use recovery line
            return "recovery" if "recovery" in self.racing_lines else "standard"
        elif drone_rank == len(ranks)-1:  
            # Last place but close - use aggressive overtaking line
            return "overtake" if "overtake" in self.racing_lines else "standard"
        else:  # Middle positions
            return "overtake" if "overtake" in self.racing_lines else "standard"
    
    def get_race_ranks(self):
        """
        Get current race rankings from the environment.
        
        Returns:
            List of drone IDs in order of race position
        """
        # Check if environment has a ranking function
        if hasattr(self.env, 'get_race_ranks'):
            return self.env.get_race_ranks()
        
        # Otherwise, calculate rankings from progress
        if not hasattr(self.env, 'vehicles'):
            return []
            
        # Sort drones by progress if available
        ranked_drones = sorted(
            self.env.vehicles.keys(),
            key=lambda drone_id: -self.get_drone_progress(drone_id)  # Higher progress first
        )
        
        return ranked_drones
    
    def get_drone_progress(self, drone_id):
        """
        Get the progress value for a drone.
        
        Args:
            drone_id: ID of the drone
            
        Returns:
            Progress value (higher is better)
        """
        # Check if environment tracks progress
        if hasattr(self.env, 'drone_progress') and drone_id in self.env.drone_progress:
            return self.env.drone_progress[drone_id]
        
        # Alternative: calculate from current target
        if hasattr(self.env, 'vehicles') and drone_id in self.env.vehicles:
            if 'current_target' in self.env.vehicles[drone_id]:
                return self.env.vehicles[drone_id]['current_target']
            elif 'current_target_index' in self.env.vehicles[drone_id]:
                return self.env.vehicles[drone_id]['current_target_index']
        
        # Default to 0 if no progress info available
        return 0
    
    def get_racing_line_direction(self, line_key, point_idx):
        """
        Get the direction vector for a specific point on a racing line.
        
        Args:
            line_key: Key for the racing line
            point_idx: Index of the point
            
        Returns:
            3D direction vector
        """
        if line_key not in self.racing_line_directions:
            return np.array([1, 0, 0])  # Default
            
        directions = self.racing_line_directions[line_key]
        if point_idx < len(directions):
            return directions[point_idx]
        else:
            return directions[0]  # Default to first direction
    
    def get_racing_line_alignment(self, drone_id, position, velocity, target_idx):
        """
        Calculate alignment between drone velocity and racing line direction.
        
        Args:
            drone_id: ID of the drone
            position: Current position of the drone
            velocity: Current velocity of the drone
            target_idx: Current target index
            
        Returns:
            Alignment value [-1, 1] where 1 means perfect alignment
        """
        # Get selected racing line
        line_key = self.select_racing_line_for_drone(drone_id)
        
        # Get current target position on the racing line
        target_position = self.get_optimal_racing_line_point(drone_id, target_idx)
        
        # Vector to target
        to_target = target_position - position
        distance = np.linalg.norm(to_target)
        
        if distance > 0:
            to_target = to_target / distance
        else:
            return 0  # At target already
        
        # Get racing line direction at this point
        racing_direction = self.get_racing_line_direction(line_key, target_idx)
        
        # Calculate gate_size for adaptive weighting
        gate_size = self.config['gate_radius']
        if hasattr(self.env, 'gate_size'):
            gate_size = self.env.gate_size
        
        # Blend target direction and racing line direction based on distance
        # When far from target, point toward target
        # When closer to target, align with racing line direction
        progress_weight = min(1.0, max(0.0, 1.0 - distance / (5 * gate_size)))
        
        desired_direction = (1 - progress_weight) * to_target + \
                           progress_weight * racing_direction
                           
        if np.linalg.norm(desired_direction) > 0:
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
        
        # Calculate alignment
        if np.linalg.norm(velocity) > 0:
            velocity_dir = velocity / np.linalg.norm(velocity)
            alignment = np.dot(velocity_dir, desired_direction)
            return alignment
        
        return 0
    
    def get_closest_point_on_racing_line(self, position, line_key="standard"):
        """
        Find the closest point on a racing line to a given position.
        
        Args:
            position: 3D position to check from
            line_key: Which racing line to use
            
        Returns:
            (closest_point, segment_idx, distance)
        """
        if line_key not in self.racing_lines:
            line_key = list(self.racing_lines.keys())[0]
            
        racing_line = self.racing_lines[line_key]
        
        closest_idx = 0
        min_distance = float('inf')
        closest_point = None
        
        # Find closest point on the racing line
        for i, point in enumerate(racing_line):
            distance = np.linalg.norm(position - point)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_point = point
        
        return closest_point, closest_idx, min_distance
    
    def update_visualization(self):
        """Update visualization markers for racing lines."""
        if not self.visualize_lines or not hasattr(self.env, 'render'):
            return
            
        # Clear previous markers
        self.visualization_markers = []
        
        # Define colors for each racing line
        colors = {
            "direct": (1, 0, 0),      # Red
            "standard": (0, 1, 0),    # Green
            "overtake": (0, 0, 1),    # Blue
            "recovery": (1, 1, 0)     # Yellow
        }
        
        # Add visualization for each line type
        for line_name, line in self.racing_lines.items():
            color = colors.get(line_name, (0.5, 0.5, 0.5))
            
            # Create markers based on environment's visualization system
            if hasattr(self.env, 'add_visualization_markers'):
                # Environment has a dedicated method
                markers = self.env.add_visualization_markers(line, color, line_width=2, closed=True)
                self.visualization_markers.extend(markers)
            elif hasattr(self.env, 'renderer') and hasattr(self.env.renderer, 'add_line'):
                # Environment has a renderer with appropriate methods
                for i in range(len(line)-1):
                    marker = self.env.renderer.add_line(line[i], line[i+1], color=color, width=2)
                    self.visualization_markers.append(marker)
                
                # Connect last to first
                marker = self.env.renderer.add_line(line[-1], line[0], color=color, width=2)
                self.visualization_markers.append(marker)