from cProfile import label

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

        self.Q = 8*np.diag([const.POS_VAR, const.POS_VAR * 10, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])
        self.R_cam = np.diag([const.POS_VAR, const.POS_VAR, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])
        self.R_nocam = np.diag([const.VEL_VAR, const.VEL_VAR])
        self.P = 500*np.diag([const.POS_VAR, const.POS_VAR, const.THETA_VAR, const.VEL_VAR, const.VEL_VAR])

    def compute_fnF(self, dt):
        theta = self.x[2]
        vf = (self.x[3] + self.x[4]) / 2
        wheel_base = 100 # mm
        
        coS = np.cos(theta)
        siN = np.sin(theta)

        f = np.array([
            [1, 0, 0, 0.5 * coS * dt, 0.5 * coS * dt],
            [0, 1, 0, 0.5 * siN * dt, 0.5 * siN * dt],
            [0, 0, 1, - dt / wheel_base, dt / wheel_base],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]]
        )

        F = np.array([
            [1, 0, -vf * siN * dt, 0.5 * coS * dt, 0.5 * coS * dt],
            [0, 1, vf * coS * dt, 0.5 * siN * dt, 0.5 * siN * dt],
            [0, 0, 1, - dt / wheel_base, dt / wheel_base],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])
        
        return f, F
    
    def prediction(self, dt):
        f, F = self.compute_fnF(dt)
        self.x = f @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, position, u, cam):
        if cam:
            R = self.R_cam
            H = self.H_cam
            z = np.array([position[0], position[1], position[2], u[0], u[1]])
        else:
            R = self.R_nocam
            H = self.H_nocam
            z = np.array([u[0], u[1]])

        z_pred = H @ self.x
        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x += K @ y
        self.P = self.P - K @ H @ self.P

    def get_state(self):
        return self.x

    def get_cov(self):
        return self.P
    
    def count_time(self, cur_t):
        self.cur_t = cur_t
        
    def previous_time(self):
        return self.cur_t