import numpy as np
import time
from ExtendedKF import KalmanFilterExtended
import ExtendedKF
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import constants as const


def confidence_ellipse(P, measured_state, ax):
    P_pos = P[:2, :2]
    eigvals, eigvecs = np.linalg.eig(P_pos)
        
    major_axis = np.sqrt(eigvals[1])
    minor_axis = np.sqrt(eigvals[0])
        
    print(f"Major Acis Length: : {2 * major_axis:.4f}, Minor Axis Length: {2 * minor_axis:.4f}")
    
    thymio_pos_x, thymio_pos_y, thymio_theta = measured_state[0], measured_state[1], measured_state[2]
    direction_angle = np.arctan2(np.sin(thymio_theta), np.cos(thymio_theta)) # deg

    ax.patches.clear()
        
    ellipse = Ellipse((thymio_pos_x, thymio_pos_y), width = 2*major_axis, height = 2*minor_axis,
                      angle = direction_angle, edgecolor = 'r', facecolor = 'none', linewidth = 2)
    ax.add_patch(ellipse)
    
    plt.pause(0.01)
    

# might also have to add kidnapping mode (i.e. kidnap = True)
def run_EKF(ekf, pos_x, pos_y, theta, u, dt = None, cam = True, ax = None):
    kidnap = False

    if dt is None:
        cur_t = time.time()
        dt = cur_t - ekf.previous_time()
        ekf.count_time(cur_t)
    
    # ekf.compute_fnF(dt)
    ekf.prediction([pos_x, pos_y, theta], u, dt)
    measured_state = ekf.get_state()
    
    if cam:
        pos_prev = np.array([measured_state[0], measured_state[1]]), 
        angle_prev = measured_state[2]
        dpos = abs(np.array([pos_x, pos_y]) - pos_prev)
        #dpos = np.linalg.norm(np.array([pos_x, pos_y]) - pos_prev)  #Euclidean distance
        dtheta = abs(theta - angle_prev)
        
        # if dpos > const.KN_DIST or dtheta > const.KN_THETA:
        #     kidnap = True
        #     print('... Thymio is being kidnapped ...')
        #     cur_t = time.time()
        #     ekf.count_time(cur_t)
        #     ekf.get_state()

    if cam:
        ekf.update(measured_state, cam)
    else:
        ekf.update(measured_state[3:], cam)
    measurement_update = ekf.get_state()
    
    if ax is not None:
        confidence_ellipse(ekf.get_cov(), measurement_update, ax)

    return measurement_update