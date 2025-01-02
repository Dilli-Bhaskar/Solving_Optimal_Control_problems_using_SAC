import numpy as np
import torch


class Pendulum:    
    def __init__(self, initial_state=None, dt=0.1, max_steps=400, noise_std=0.01):
        # Initialize parameters for the TWSBR model
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        self.noise_std = noise_std

        # Robot physical parameters
        self.Mc = 21  # Chassis mass (kg)
        self.mw = 0.42  # Wheel mass (kg)
        self.r = 0.106  # Wheel radius (m)
        self.l = 0.3  # Distance to the center of mass (m)
        self.d = 0.44  # Distance between wheels (m)
        self.Jz = 0.63  # Moment of inertia (kg*m^2)
        self.Jw = 0.0024
        self.Jy = 0.3388
        self.g = 9.81
        
        # State-space model matrices (subsystem 1 and 2)
        self.A1 = np.array([[1, 0.1, 0.2955, 0.01043],
                            [0, 1, 5.073, 0.2955],
                            [0, 0, 0.2816, 0.07465],
                            [0, 0, -12.33, 0.2816]])
        self.B1 = np.array([[0.007831],
                            [0.1314],
                            [-0.02165],
                            [-0.3717]])

        self.A2 = np.array([[1, 0.1],
                            [0, 1]])
        self.B2 = np.array([[0.02596],
                            [0.5192]])

        # Initial state
        self.state_dim1 = 4  # Subsystem 1 state dimensions
        self.state_dim2 = 2  # Subsystem 2 state dimensions
        self.initial_state = initial_state if initial_state is not None else np.hstack([
            np.array([0.1] * self.state_dim1),  # Subsystem 1 initial state
            np.array([0.1] * self.state_dim2)   # Subsystem 2 initial state
        ])
        self.state = self.initial_state.copy()
                
        self.state_dim = 6
        self.action_dim = 2
        self.action_max = np.array([1, 1])
        self.action_min = np.array([-1, -1])
        self.terminal_time = 40
        self.dt = 0.1
        self.initial_state = np.hstack([np.array([0.1] * 4), np.array([0.1] * 2)])
        self.inner_step_n = 2  # Example value
        self.inner_dt = 0.05   # Example value
        self.gravity = 9.8
        self.r = 0.01
        self.beta = self.r
        self.m = 1.
        self.l = 1.


    def reset(self):
        """Reset the environment to the initial state."""
        self.state = self.initial_state.copy()
        self.current_step = 0
        return self.state

    def step(self, action):
        """Perform one time step in the environment."""
        u1, u2 = action[0], action[1]

        # Add noise to the action to ensure persistence of excitation
        u1 += np.random.normal(0, self.noise_std)
        u2 += np.random.normal(0, self.noise_std)

        # Subsystem 1 dynamics
        x1 = self.state[:self.state_dim1]
        x1_next = self.A1 @ x1 + self.B1.flatten() * u1

        # Subsystem 2 dynamics
        x2 = self.state[self.state_dim1:]
        x2_next = self.A2 @ x2 + self.B2.flatten() * u2

        # Update the state
        self.state = np.hstack([x1_next, x2_next])
        self.current_step += 1

        # Define reward (example: penalize deviation from zero state and control effort)
        reward = - (np.linalg.norm(self.state) ** 2 + 0.1 * (u1 ** 2 + u2 ** 2))

        # Termination condition
        done = self.current_step >= self.max_steps

        return self.state, reward, done, None


    def get_state_obs(self):
        return (
        f"x: {self.state[0]:.3f} m, "
        f"v: {self.state[1]:.3f} m/s, "
        f"θ: {self.state[2]:.3f} rad, "
        f"ω: {self.state[3]:.3f} rad/s, "
        f"ψ: {self.state[4]:.3f} rad, "
        f"ψ̇: {self.state[5]:.3f} rad/s"
    )
 
    def render(self):
        """Render the environment state."""
        print(f"State: {self.state}")
