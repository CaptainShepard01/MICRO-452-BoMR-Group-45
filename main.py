import math
import os
from copy import deepcopy
import time

import cv2
import numpy as np

from thymio import Thymio
from ComputerVision import ComputerVision
from ExtendedKF import KalmanFilterExtended
from global_navigation import Navigation


def get_frame_with_vectors(vision, frame):
    frame_with_markers, marker_ids, rvecs, tvecs, aruco_side_pixels = vision.detect_and_estimate_pose(frame)

    # Will contain ID, x, y, angle of each detected marker
    markers_data = []

    if marker_ids is not None:
        frame_with_vectors, markers_data = vision.process_marker_pose(frame_with_markers, marker_ids, rvecs, tvecs)
    else:
        frame_with_vectors = frame.copy()
    
    # Conversion from pixels to mm
    conversion_factor = aruco_side_pixels/45

    return frame_with_vectors, markers_data, marker_ids, conversion_factor

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

    corners = []

    ret, frame = vision.cam.read()
    # While loop here to make sure camera is not noisy
    while True:
        if not ret:
            print("Failed to capture frame. Exiting.")

        frame_masked = vision.apply_color_mask(frame,
                                               mask_color='r')  # We apply a red mask to detect only red obstacles
        edges = vision.detect_edges(frame_masked)  # We detect the edges on the masked frame using Canny
        corners = vision.get_corners_and_shape_edges(edges)  # We obtain shapes and corners by approxPolyDP
        corners_mm = [np.array(shape) * 1/conversion_factor for shape in corners]

        ret, frame = vision.cam.read()

        if len(corners) == 5:
            break

    frame_with_vectors, markers_data, marker_ids = get_frame_with_vectors(vision, frame)
    thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)
    goal_pos = get_goal_position(markers_data)

    # Global path planning
    # navigation = Navigation(corners, [thymio_pos_x, thymio_pos_y], goal_pos)
    # global_path = navigation.get_shortest_path()

    thymio.set_position(np.array([thymio_pos_x, thymio_pos_y]))
    thymio.set_orientation(thymio_theta)

    # create and initialize the Kalman filter
    # u = thymio.get_wheels_speed()
    ekf = KalmanFilterExtended(np.array([thymio_pos_x, thymio_pos_y, thymio_theta]), [0, 0])

    while not thymio.is_on_goal:
    # while True:
        # local navigation
        # if np.any(thymio.get_horizontal_sensors() > thymio.OBSTACLE_THRESHOLD):
        #     thymio.local_navigation()

        ### Detection and position of the markers (ROBOT and GOAL) ###
        ret, frame = vision.cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        ### Detection of red edges and red corners ###
        frame_masked = vision.apply_color_mask(frame, mask_color='r')   # We apply a red mask to detect only red obstacles
        edges = vision.detect_edges(frame_masked)                       # We detect the edges on the masked frame using Canny
        corners = vision.get_corners_and_shape_edges(edges)             # We obtain shapes and corners by approxPolyDP
        corners_mm = [np.array(shape) * 1/conversion_factor for shape in corners]

        frame_with_vectors, markers_data, marker_ids = get_frame_with_vectors(vision, frame)
        frame_aruco_and_corners = frame_with_vectors.copy()

        thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)
        thymio.set_position(np.array([thymio_pos_x, thymio_pos_y]))
        thymio.set_orientation(thymio_theta)

        # Visualize corners
        for shape in corners:
            for x, y in shape:
                cv2.circle(frame_aruco_and_corners, (x, y), 3, (0, 255, 0), -1)

        # prev_point = global_path[0]
        # for point in global_path:
        #     cv2.circle(frame_aruco_and_corners, (int(point.x), int(point.x)), 5, (0, 0, 255), -1)
        #     # cv2.line(frame_aruco_and_corners, [int(prev_point.x), int(prev_point.y)], [int(point.x), int(point.y)], (255, 0, 0), 2)
        #     prev_point = point

        cv2.imshow("Main", frame_aruco_and_corners)

        # print("Shapes:")
        # for shape in init_corners:
        #     print(len(shape))
        

        ### Detection if camera is covered ###
        # We check if self.ARUCO_ROBOT_ID is in the detected markers
        camera_covered = vision.is_camera_covered(marker_ids)
        # print(f"Robot not detected: {camera_covered} (ID robot = {vision.ARUCO_ROBOT_ID})")

        # Kalman filter

        # movement
        thymio.move_to_point_astolfi(np.array(goal_pos))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()
    thymio.stop()
    print("Goal reached")