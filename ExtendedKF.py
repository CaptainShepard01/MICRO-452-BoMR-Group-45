import numpy as np
import time
import constants as const
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
   
class KalmanFilterExtended:
    
    def __init__(self, initial_pos, u): 
        self.cur_t = 0
        self.count_time(time.time())
        self.x = np.zeros(5)
        self.x[3:] = u
        self.x[:3] = initial_pos
        
        self.H_cam = np.eye(5)
        self.H_nocam = np.array([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1]])

        self.Q = 8*np.diag([const.POS_VAR, const.POS_VAR, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])
        self.R_cam = np.diag([const.POS_VAR, const.POS_VAR, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])
        self.R_nocam = np.diag([const.VEL_VAR, const.VEL_VAR])
        self.P = 8*np.diag([const.POS_VAR, const.POS_VAR, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])
        
    def compute_fnF(self, dt):
        # pos_x, pos_y, theta = self.x[0], self.x[1], self.x[2]
        # left_wheel_speed, right_wheel_speed = self.x[3], self.x[4]
        theta = self.x[2]
        avg_v = (self.x[3] + self.x[4]) / 2
        wheel_base = 100 # mm 
        omega = (self.x[3] - self.x[4]) / wheel_base
        vf = avg_v * omega
        coS = np.cos(theta)
        siN = np.sin(theta)
        
        f = np.array([
            [1, 0, 0, - 0.5 * coS * dt, - 0.5 * coS * dt],
            [0, 1, 0, 0.5 * siN * dt, 0.5 * siN * dt],
            [0, 0, 1, - 0.5 * dt / wheel_base, 0.5 * dt / wheel_base],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])
                     
        F = np.array([
            [1, 0, vf * siN * dt, - 0.5 * coS * dt, - 0.5 * coS * dt],
            [0, 1, vf * coS * dt, 0.5 * siN * dt, 0.5 * siN * dt],
            [0, 0, 1, - 0.5 * dt / wheel_base, 0.5 * dt / wheel_base],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])
        
        return f, F
    
    def prediction(self, dt):
        f, F = self.compute_fnF(dt)
        self.x = f @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  
        self.P += np.eye(self.P.shape[0]) * 1e-6 
    
    def update(self, z, cam, learning_rate=0.1):
        if cam:
            R = self.R_cam
            H = self.H_cam
        else:
            R = self.R_nocam
            H = self.H_nocam
        
        z_pred = H @ self.x
        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        # Additional covariance update with residual outer product
        update = learning_rate * np.outer(y, y)
        if H.shape[0] == 2:  # 2D measurement space (e.g., position)
            self.P[:2, :2] += update  # Update only the top-left 2x2 block
        else:
            self.P += update
        # Ensure covariance matrix symmetry and numerical stability
        self.P = 0.5 * (self.P + self.P.T) + np.eye(self.P.shape[0]) * 1e-6

    def get_state(self):
        return self.x

    def get_cov(self):
        return self.P
    
    def count_time(self, cur_t):
        self.cur_t = cur_t
        
    def previous_time(self):
        return self.cur_t