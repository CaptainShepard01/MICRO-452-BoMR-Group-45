import math
import os

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
    thymio_pos_x, thymio_pos_y, thymio_theta = None, None, None

    for marker in markers_data:
        if marker[0] == vision.ARUCO_ROBOT_ID:
            thymio_pos_x, thymio_pos_y = marker[1], marker[2]
            thymio_theta = marker[3]

    return thymio_pos_x, thymio_pos_y, thymio_theta

def get_goal_position(markers_data):
    goal_pos = None

    for marker in markers_data:
        if marker[0] == vision.ARUCO_GOAL_ID:
            goal_pos = [marker[1], marker[2]]
            break

    return goal_pos

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def merge_close_corners_simple(corners, threshold: float = 5):
    def merge_shape(shape):
        unique_points = []
        for point in shape:
            if not any(distance(point, unique_point) < threshold for unique_point in unique_points):
                unique_points.append(point)
        return unique_points

    return [merge_shape(shape) for shape in corners]


if __name__ == "__main__":
    vision = ComputerVision(1)
    thymio = Thymio()
    thymio.reset_leds()

    # Detecting initial Thymio position
    ret = None
    frame = None
    while ret is None:
        ret, frame = vision.cam.read()

    frame_with_vectors, markers_data, marker_ids = get_frame_with_vectors(vision, frame)

    frame_masked = vision.apply_color_mask(frame, mask_color='r')  # We apply a red mask to detect only red obstacles
    edges = vision.detect_edges(frame_masked)  # We detect the edges on the masked frame using Canny
    corners = vision.get_corners_and_shape_edges(edges)  # We obtain shapes and corners by approxPolyDP
    corners = merge_close_corners_simple(corners, threshold=5)
    corners = [corner for corner in corners if len(corner) > 2]

    thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)
    goal_pos = get_goal_position(markers_data)

    # Global path planning
    navigation = Navigation(corners, [thymio_pos_x, thymio_pos_y], goal_pos)
    global_path = navigation.get_shortest_path()

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
                cv2.circle(frame_aruco_and_corners, (x, y), 2, (0, 255, 0), -1)

        # for point in global_path:
        #     cv2.circle(frame_aruco_and_corners, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        cv2.imshow("Main", frame_aruco_and_corners)
        

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