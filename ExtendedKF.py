import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class KalmanFilterExtended:
    POS_VAR = 10 
    VEL_VAR = 10
    THETA_VAR = 10
    
    def __init__(self, initial_pos, u = np.array([0, 0])):  # Assumed initial_pos = [pos_x, pos_y, theta]
        self.x = np.zeros(5)
        self.x[3:] = u
        self.x[:3] = initial_pos
        
        self.H_cam = np.eye(5)
        self.H_nocam = np.array([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1]])

        self.Q = np.eye(5) * 0.0001
        self.R_cam = np.eye(5) * 0.01
        self.R_nocam = np.eye(2) * 0.05
        self.P = np.eye(5) * 0.01
    
    def compute_fnF(self, dt):
        pos_x, pos_y, theta = self.x[0], self.x[1], self.x[2]
        left_wheel_speed, right_wheel_speed = self.x[3], self.x[4]
        v = (self.x[3] + self.x[4]) / 2
        omega = (self.x[3] - self.x[4]) / 10  # 10 cm
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt
        
        f = np.array([
            pos_x + dx,  # predicted x position
            pos_y + dy,  # predicted y position
            theta + dtheta,  # predicted theta (orientation)
            left_wheel_speed,  # left wheel speed remains the same
            right_wheel_speed  # light wheel speed remains the same
            ])
        
        coS = np.cos(theta)
        siN = np.sin(theta)
        
        F = np.array([
             [1, 0, -v * siN * dt, 0.5 * coS * dt, 0.5 * coS * dt],
             [0, 1, v * coS * dt, 0.5 * siN * dt, 0.5 * siN * dt],
             [0, 0, 1, -dt / 10, dt / 10],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]])
        
        return f, F
        
        
    def prediction(self, dt):
        f, F = self.compute_fnF(dt)
        self.x = f
        self.P = F @ self.P @ F.T + self.Q

        # Enforce symmetry and regularization for numerical stability
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
        self.P += np.eye(self.P.shape[0]) * 1e-6  # Add small diagonal term for stability
        
    def update(self, z, cam):
        if cam:
            R = self.R_cam
            H = self.H_cam
        else:
            R = self.R_nocam
            H = self.H_nocam
        
        z_pred = H @ self.x 
        z = np.asarray(z)

        # Innovation
        y = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x += K @ y

        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P


    def get_state(self):
        return self.x

    def get_cov(self):
        return self.P

