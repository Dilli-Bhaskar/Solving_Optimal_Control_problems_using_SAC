import numpy as np
import torch


class RotationalPendulum2DOF:
    def __init__(self, initial_state=np.array([0, np.pi, 0, 0]), dt=0.2, terminal_time=5, inner_step_n=2,
                 action_min=np.array([-2]), action_max=np.array([2])):
        # State: [time, angle1 (pendulum), angular_velocity1 (pendulum), angle2 (base), angular_velocity2 (base)]
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

        # Parameters
        self.gravity = 9.8
        self.r = 0.01  # damping coefficient
        self.beta = self.r
        self.m = 1.0  # mass of the pendulum
        self.M = 1.0  # mass of the base
        self.l = 1.0  # length of the pendulum
        self.L = 1.0  # distance to the center of mass of the base
        self.state = self.initial_state

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n

    def g(self, state):
        # Gravity force calculation, can be adjusted for 2-DOF system
        return torch.stack([torch.zeros(state.shape[1]), 
                            torch.ones(state.shape[1]) * 3 / (self.m * self.l ** 2)]).transpose(0, 1).unsqueeze(1).type(torch.FloatTensor)

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            # Extract state components
            time, theta1, theta2, omega1, omega2 = self.state

            # Equations of motion for the 2-DOF system
            dtheta1 = omega1
            dtheta2 = omega2
            domega1 = (-self.m * self.l * self.gravity * np.sin(theta1) + self.M * self.L * omega2 ** 2 * np.sin(theta2 - theta1)
                      - 3 * self.gravity * np.sin(theta1)) / (self.m * self.l ** 2)
            domega2 = (-self.m * self.l * self.gravity * np.sin(theta2) + self.m * self.l * omega1 ** 2 * np.sin(theta1 - theta2)
                      + 3 * self.gravity * np.sin(theta2)) / (self.M * self.L ** 2)

            # Update state
            self.state = np.array([time + self.inner_dt, theta1 + dtheta1 * self.inner_dt, theta2 + dtheta2 * self.inner_dt,
                                   omega1 + domega1 * self.inner_dt, omega2 + domega2 * self.inner_dt])

        # Reward and done conditions
        if self.state[0] >= self.terminal_time:
            reward = - np.abs(self.state[1]) - 0.1 * np.abs(self.state[2])  # Penalize angle deviations
            done = True
        else:
            reward = - self.r * (action[0] ** 2) * self.dt  # Penalize action magnitude
            done = False

        return self.state, reward, done, None

    def get_state_obs(self):
        return 'time: %.3f  angle1: %.3f  omega1: %.3f  angle2: %.3f  omega2: %.3f' % (
            self.state[0], self.state[1], self.state[3], self.state[2], self.state[4])

    def render(self):
        print(self.get_state_obs())

