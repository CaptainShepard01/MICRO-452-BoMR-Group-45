import numpy as np
import time
from ExtendedKF import KalmanFilterExtended
import ExtendedKF

# might also have to add kidnapping mode (i.e. kidnap = True)
def run_EKF(pos_x, pos_y, theta, left_wheel_speed, right_wheel_speed, dt = None, cam = True): 
    kidnap = False
    initial_pos = np.array([pos_x, pos_y, theta])
    u = np.array([left_wheel_speed, right_wheel_speed])
    ekf = KalmanFilterExtended(initial_pos, u)

    if dt is None:
        cur_t = time.time()
        dt = cur_t - ekf.previous_time()
        ekf.count_time(cur_t)
    
    ekf.compute_fnF(dt)
    ekf.prediction(dt)
    measured_pos = ekf.get_state()
    
    if cam = True:
        pos_prev = np.array([measured_pos[0], measured_pos[1]]), 
        angle_prev = measured_pos[2]
        dpos = abs(np.array([pos_x, pos_y]) - pos_prev)
        dtheta = abs(theta - angle_prev)
        
        if dpos > ExtendedKF.KN_DIST | dtheta > ExtendedKF.KN_THETA:
            kidnap = True
            print('... Thymio is being kidnapped ...')
            cur_t = time.time()
            ekf.count_time(cur_t)
            ekf.get_state()
        
        ekf.update(measured_pos, cam,  0.1)
      
    ekf.update(measured_pos, cam,  0.1) 
    measurement_update = ekf.get_state()

    return measurement_update