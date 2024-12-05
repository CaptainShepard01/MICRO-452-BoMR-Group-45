import numpy as np
import matplotlib.pyplot as plt
from ExtendedKF import KalmanFilterExtended
from run_EKF import run_EKF
from matplotlib.patches import Ellipse

def draw_confidence_ellipse(ax, mean, cov, n_std=1.0, edgecolor='blue', alpha=0.3):
    eigvals, eigvecs = np.linalg.eig(cov)
    largest_eigval_idx = np.argmax(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, largest_eigval_idx], eigvecs[0, largest_eigval_idx]))
    width, height = 2 * n_std * np.sqrt(eigvals)  
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        edgecolor=edgecolor, facecolor=edgecolor, alpha=alpha, linestyle='--'
    )
    ax.add_patch(ellipse)

#######################################

true_path = np.array([
    (50.00, 430.00),
    (52.96, 398.67),
    (55.92, 367.33),
    (58.88, 333.02),
    (97.36, 305.53),
    (135.84, 278.05),
    (173.32, 250.54),
    (298.88, 183.69),
    (424.44, 116.85),
    (550.00, 50.00)
])

x_ini = np.array([50.00, 430.00, 0])
dt = 0.1
vl = np.array([20, 28, 70, 100, 150, 230, 280, 340, 280, 100])
vr = vl
pos = np.zeros((len(true_path), 3))

ekf = KalmanFilterExtended(x_ini[:3], np.array([vl[0], vr[0]])) 
vl_withNoise = vl + np.random.normal(0, 3, len(vl)) 
vr_withNoise = vr + np.random.normal(0, 3, len(vr)) 

pos_withNoise = np.copy(true_path) 
pos_withNoise[1:, 0] += np.random.normal(0, 3, len(true_path)-1)
pos_withNoise[1:, 1] += np.random.normal(0, 3, len(true_path)-1)

filtered_path_cam = np.zeros_like(pos)
filtered_path_nocam = np.zeros_like(pos)
cov_matrices_cam = []
cov_matrices_nocam = [] 

for i in range(len(true_path)):
    z_wheel = np.array([vl_withNoise[i], vr_withNoise[i]]) 
    
    measured_state_cam = run_EKF(ekf, pos_withNoise[i, 0], pos_withNoise[i, 1], 0, z_wheel, dt=dt, cam=True)
    filtered_path_cam[i] = measured_state_cam[:3]
    
    cov_matrix_cam = ekf.get_cov()[:2, :2] 
    cov_matrices_cam.append(cov_matrix_cam)
    
    measured_state_nocam = run_EKF(ekf, pos_withNoise[i, 0], pos_withNoise[i, 1], 0, z_wheel, dt=dt, cam=False)
    filtered_path_nocam[i] = measured_state_nocam[:3] 

    cov_matrix_nocam = ekf.get_cov()[:2, :2] 
    cov_matrices_nocam.append(cov_matrix_nocam) 

a1_width = 841 
a1_height = 594  
max_x = max(true_path[:, 0])
max_y = max(true_path[:, 1])

scale_x = a1_width / max_x
scale_y = a1_height / max_y

scaled_true_path = true_path * np.array([scale_x, scale_y])
scaled_pos_withNoise = pos_withNoise * np.array([scale_x, scale_y])
scaled_filtered_path_cam = filtered_path_cam[:, :2] * np.array([scale_x, scale_y])
scaled_filtered_path_nocam = filtered_path_nocam[:, :2] * np.array([scale_x, scale_y])

obstacles = [
    [[147, 358], [83, 358], [81, 415], [146, 414]], 
    [[547, 291], [428, 329], [549, 360]],  
    [[402, 184], [213, 260], [299, 358]],  
    [[77, 154], [78, 208], [141, 208], [140, 152]],  
    [[297, 60], [300, 112], [359, 111], [356, 60]]  
]

plt.figure(figsize=(10, 10))
ax = plt.gca()

for obstacle in obstacles:
    obstacle = np.array(obstacle) * np.array([scale_x, scale_y])
    obstacle = np.vstack([obstacle, obstacle[0]]) 
    plt.plot(obstacle[:, 0], obstacle[:, 1], color="brown", linestyle="-", linewidth=2.2)

for i in range(len(true_path)):
    mean_cam = scaled_filtered_path_cam[i, :2]
    cov_matrix_cam = cov_matrices_cam[i] 
    draw_confidence_ellipse(ax, mean_cam, 150 * cov_matrix_cam, n_std=2.0, edgecolor='blue', alpha=0.3)

    mean_nocam = scaled_filtered_path_nocam[i, :2]  
    cov_matrix_nocam = cov_matrices_nocam[i] 
    draw_confidence_ellipse(ax, mean_nocam, 50 * cov_matrix_nocam, n_std=2.0, edgecolor='orange', alpha=0.3)


plt.plot(scaled_true_path[0, 0], scaled_true_path[0, 1], marker='P', markersize=15, color="brown", label="Start")
plt.plot(scaled_true_path[-1, 0], scaled_true_path[-1, 1], marker='X', markersize=15, color="red", label="Goal")
plt.plot(scaled_true_path[:, 0], scaled_true_path[:, 1], linestyle="-", marker='o', markersize=8, color="black", label="True path")
plt.plot(scaled_pos_withNoise[:, 0], scaled_pos_withNoise[:, 1], linestyle="--", marker='*', markersize=15, color="darkgreen", label="Position from camera")
plt.plot(scaled_filtered_path_cam[:, 0], scaled_filtered_path_cam[:, 1], linestyle="--", marker='s', markersize=6, color="deepskyblue", label="Filtered position (with vision)")
plt.plot(scaled_filtered_path_nocam[:, 0], scaled_filtered_path_nocam[:, 1], linestyle="--", marker='^', markersize=6, color="purple", label="Filtered position (without vision)")
plt.legend()
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Extended Kalman Filter Simulation")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, a1_width + 40)
plt.ylim(0, a1_height + 40)
plt.show()
