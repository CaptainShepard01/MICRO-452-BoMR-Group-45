import numpy as np
import time
from ExtendedKF import KalmanFilterExtended
import ExtendedKF
import matplotlib.pyplot as plt
import constants as const

def run_EKF(ekf, pos_x, pos_y, theta, u, dt = None, cam = True, ax = None):
    kidnap = False

    if dt is None:
        cur_t = time.time()
        dt = cur_t - ekf.previous_time()
        ekf.count_time(cur_t)

    ekf.prediction(dt)
    measured_state = ekf.get_state()
    
    if cam:
        pos_prev = np.array([measured_state[0], measured_state[1]])
        if pos_prev is not None:
            angle_prev = measured_state[2]
            dpos = np.sqrt((pos_x - pos_prev[0])**2 + (pos_y - pos_prev[1])**2)
            dtheta = abs(theta - angle_prev)

            if dpos > const.KN_DIST or dtheta > const.KN_THETA:
                kidnap = True
                print("Thymio is being kidnapped (moved too far or rotated too much)")
                cur_t = time.time()
                ekf.count_time(cur_t)
                ekf.get_state()
            else:
                kidnap = False
    else:
        kidnap = False

    ekf.update([pos_x, pos_y, theta], u, cam)
    measurement_update = ekf.get_state()

    return measurement_update, kidnap