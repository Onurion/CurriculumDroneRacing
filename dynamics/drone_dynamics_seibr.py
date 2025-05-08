import numpy as np


# -------------------------------
# Drone Dynamics Model (12-State)
# -------------------------------
class DroneDynamics:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.mass = 1.0  # kg

        # Inertia matrix (assumed diagonal)
        self.Ixx = 0.005  # kg*m^2
        self.Iyy = 0.005  # kg*m^2
        self.Izz = 0.009  # kg*m^2

        # Gravity acceleration (m/s^2)
        self.g = 9.81

        # State vector [ x, y, z, vx, vy, vz, phi, theta, psi, p, q, r ]
        self.state = np.zeros(12, dtype=np.float32)
        self.state[2] = 0.0         # Starting at zero altitude
        self.state[6:9] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def rotation_matrix(self, phi, theta, psi):
        # Rotation using ZYX Euler angles (body-to-inertial)
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                       [np.sin(psi),  np.cos(psi), 0.0],
                       [0.0,         0.0,         1.0]])
        Ry = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                       [0.0,           1.0, 0.0],
                       [-np.sin(theta),0.0, np.cos(theta)]])
        Rx = np.array([[1.0,  0.0,           0.0],
                       [0.0,  np.cos(phi), -np.sin(phi)],
                       [0.0,  np.sin(phi),  np.cos(phi)]])
        return Rz @ Ry @ Rx

    def euler_derivatives(self, phi, theta, psi, p, q, r):
        # Compute derivatives of Euler angles given the body angular velocities.
        cos_theta = np.cos(theta) if abs(np.cos(theta)) > 1e-6 else 1e-6
        phi_dot   = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot   = q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta
        return np.array([phi_dot, theta_dot, psi_dot], dtype=np.float32)

    def dynamics(self, state, control):
        """
        Compute state derivative.
          state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
          control: [T, tau_x, tau_y, tau_z]
        """
        # Unpack state
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        T, tau_x, tau_y, tau_z = control

        # Thrust in body frame and gravity
        R = self.rotation_matrix(phi, theta, psi)
        thrust_body = np.array([0.0, 0.0, T])
        thrust_inertial = R @ thrust_body
        gravity = np.array([0.0, 0.0, -self.mass * self.g])
        acceleration = (thrust_inertial + gravity) / self.mass

        # Euler angles derivative (from body ang. velocities)
        euler_dot = self.euler_derivatives(phi, theta, psi, p, q, r)

        # Angular dynamics: Euler’s equations (assuming a diagonal inertia matrix)
        dp = (tau_x - (self.Izz - self.Iyy) * q * r) / self.Ixx
        dq = (tau_y - (self.Ixx - self.Izz) * p * r) / self.Iyy
        dr = (tau_z - (self.Iyy - self.Ixx) * p * q) / self.Izz

        # Pack state derivative
        state_dot = np.zeros(12, dtype=np.float32)
        state_dot[0:3] = [vx, vy, vz]
        state_dot[3:6] = acceleration
        state_dot[6:9] = euler_dot
        state_dot[9:12] = [dp, dq, dr]
        return state_dot

    def step(self, control):
        """
        Advance the state using Euler integration.
        """
        state_dot = self.dynamics(self.state, control)
        self.state = self.state + state_dot * self.dt
        self.state[2] = max(0.0, self.state[2])  # Ensure altitude is non-negative
        # Wrap the Euler angles (phi, theta, psi)
        self.state[6] = wrap_angle(self.state[6])
        self.state[7] = wrap_angle(self.state[7])
        self.state[8] = wrap_angle(self.state[8])
        
        return self.state, state_dot

    def reset(self, pos):
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = pos   # position
        self.state[6:9] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return self.state




# -------------------------------
# Low-Level PD Controller
# -------------------------------
class LowLevelController:
    """
    Low-level controller converting the desired acceleration command (provided by RL)
    into control inputs for the drone dynamics.
    
    The controller uses PD feedback on the drone's velocity (and implicitly on the acceleration
    error) to compute corrective inputs.
    
    The PD gains (Kp_pos and Kd_pos) are the same as those determined during PD tuning.
    Additionally, a simple inner-loop for attitude stabilization is implemented.
    """
    def __init__(self, drone,
                 Kp_pos=np.array([1.0, 1.0, 1.0]),  # Tuned proportional gains for acceleration tracking
                 Kd_pos=np.array([0.5, 0.5, 0.5]),  # Tuned derivative gains for velocity damping
                 Kp_att=np.array([0.1, 0.1, 0.1]),  # Attitude (angle) proportional gains
                 Kd_att=np.array([0.05, 0.05, 0.05]) # Attitude (angular velocity) derivative gains
                ):
        self.drone = drone
        self.Kp_pos = Kp_pos
        self.Kd_pos = Kd_pos
        
        self.Kp_att = Kp_att
        self.Kd_att = Kd_att

    def compute_control(self, desired_acc):
        """
        Compute control commands from desired acceleration by combining a feedforward
        component (the RL command) with a PD feedback correction based on the current velocity.
        
        The control command includes:
          T: Total thrust (N)
          tau_x, tau_y, tau_z: Torques to be applied about the body axes.
        """
        # Current drone state (position, velocity, and attitude)
        state = self.drone.state
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]  # [phi, theta, psi]

        # Compute the error.
        # Even though desired_acc is provided by the RL agent (feedforward), we add
        # a feedback term (PD correction) to ensure accurate tracking.
        # Here, pos_error is interpreted as: "How much acceleration do we want beyond what we have?"
        acc_command = self.Kp_pos * desired_acc - self.Kd_pos * vel

        # Compute the required total thrust:
        # To hover, thrust should counteract gravity.
        T_hover = self.drone.mass * self.drone.g
        T = T_hover + self.drone.mass * acc_command[2]

        # For lateral components, compute approximate desired roll (phi_des) and pitch (theta_des)
        # These are rough approximations derived from the acceleration commands.
        desired_phi = -acc_command[1] / self.drone.g   # roll: negative sign for coordinate consistency
        desired_theta = acc_command[0] / self.drone.g    # pitch

        # Attitude error: compare desired vs. current angles:
        desired_angles = np.array([desired_phi, desired_theta, 0.0])
        error_angles = desired_angles - angles

        # PD control on attitude (using current angular velocities)
        angular_rates = state[9:12]
        tau = self.Kp_att * error_angles - self.Kd_att * angular_rates

        control = np.array([T, tau[0], tau[1], tau[2]])
        return control
    

class VelocityTracker:
    """
    Outer loop velocity tracker: takes a velocity reference and produces
    a desired acceleration command that is then fed into the low-level controller.
    """
    def __init__(self, drone, low_level_controller,
                 K_v=np.array([1.0, 1.0, 1.0]),
                 K_vd=np.array([1.0, 1.0, 1.0])):  # Velocity tracking gains (tunable)
        self.drone = drone
        self.low_level_controller = low_level_controller
        self.K_v = K_v
        self.K_vd = K_vd
        self.prev_error = 0.0

    def compute_control(self, v_ref):
        """
        Compute the overall control command given a velocity reference.

        v_ref: desired velocity vector (x, y, z)
        
        Steps:
         1. Compute the error between desired and actual velocity.
         2. Map that error to a desired acceleration command.
         3. Pass the acceleration command to the low-level controller.
        """
        # Extract current velocity (vx, vy, vz) from the drone's state.
        current_vel = self.drone.state[3:6]
        vel_error = (v_ref - current_vel)
        vel_error_d = vel_error - self.prev_error
        # Simple PD-like outer loop (here, derivative is implicitly managed by low-level feedback)
        desired_acc = self.K_v * vel_error + self.K_vd * vel_error_d
        # Compute the final control command using the low-level controller.
        control = self.low_level_controller.compute_control(desired_acc)
        self.prev_error = vel_error
        return control

def wrap_angle(angle):
    """
    Wrap an angle (in radians) to the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi