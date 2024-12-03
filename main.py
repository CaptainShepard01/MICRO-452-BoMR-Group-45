import cv2
import numpy as np

from thymio import Thymio
from ComputerVision import ComputerVision
from ExtendedKF import KalmanFilterExtended
from global_navigation import Navigation


def get_frame_with_vectors(vision, frame):
    frame_with_markers, marker_ids, rvecs, tvecs = vision.detect_and_estimate_pose(frame)

    # Will contain ID, x, y, angle of each detected marker
    markers_data = []

    if marker_ids is not None:
        frame_with_vectors, markers_data = vision.process_marker_pose(frame_with_markers, marker_ids, rvecs, tvecs)
    else:
        frame_with_vectors = frame.copy()

    return frame_with_vectors, markers_data, marker_ids

def get_thymio_localisation(markers_data):
    for marker in markers_data:
        if marker[0] == vision.ARUCO_ROBOT_ID:
            thymio_pos_x, thymio_pos_y = marker[1], marker[2]
            thymio_theta = marker[3]

    return thymio_pos_x, thymio_pos_y, thymio_theta

def get_goal_position(markers_data):
    for marker in markers_data:
        if marker[0] == vision.ARUCO_GOAL_ID:
            goal_pos = [marker[1], marker[2]]
            break

    return goal_pos


if __name__ == "__main__":
    vision = ComputerVision(1)

    # Detecting initial Thymio position
    ret = None
    frame = None
    while ret is None:
        ret, frame = vision.cam.read()

    frame_with_vectors, markers_data, marker_ids = get_frame_with_vectors(vision, frame)
    thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)

    frame_masked = vision.apply_color_mask(frame, mask_color='r')  # We apply a red mask to detect only red obstacles
    edges = vision.detect_edges(frame_masked)  # We detect the edges on the masked frame using Canny
    corners = vision.get_corners_and_shape_edges(edges)  # We obtain shapes and corners by approxPolyDP
    goal_pos = get_goal_position(markers_data)

    # Global path planning
    navigation = Navigation(corners, [thymio_pos_x, thymio_pos_y], goal_pos)
    global_path = navigation.get_shortest_path()

    # create and initialize the Thymio
    thymio = Thymio()
    thymio.set_position(np.array([thymio_pos_x, thymio_pos_y]))
    thymio.set_orientation(thymio_theta)

    print(f"Initial position: {thymio_pos_x, thymio_pos_y, thymio_theta}")

    # create and initialize the Kalman filter
    u = thymio.get_wheels_speed()
    ekf = KalmanFilterExtended(np.array([thymio_pos_x, thymio_pos_y, thymio_theta]), u)

    while not thymio.is_on_goal:
        # local navigation
        # if np.any(thymio.get_horizontal_sensors() > thymio.OBSTACLE_THRESHOLD):
        #     thymio.local_navigation()

        ### Detection and position of the markers (ROBOT and GOAL) ###
        ret, frame = vision.cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        frame_with_vectors, markers_data, marker_ids = get_frame_with_vectors(vision, frame)

        ### Detection of red edges and red corners ###
        frame_masked = vision.apply_color_mask(frame, mask_color='r')   # We apply a red mask to detect only red obstacles
        edges = vision.detect_edges(frame_masked)                       # We detect the edges on the masked frame using Canny
        corners = vision.get_corners_and_shape_edges(edges)             # We obtain shapes and corners by approxPolyDP

        frame_aruco_and_corners = frame_with_vectors.copy()

        # Visualize corners
        for shape in corners:
            for x, y in shape:
                cv2.circle(frame_aruco_and_corners, (x, y), 5, (0, 255, 0), -1)
        

        ### Detection if camera is covered ###
        # We check if self.ARUCO_ROBOT_ID is in the detected markers
        camera_covered = vision.is_camera_covered(marker_ids)
        print(f"Robot not detected: {camera_covered} (ID robot = {vision.ARUCO_ROBOT_ID})")

        # Kalman filter

        # movement

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()
    thymio.stop()
    print("Goal reached")