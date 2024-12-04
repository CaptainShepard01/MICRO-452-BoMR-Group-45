import math
import os
from copy import deepcopy
import time

import cv2
import numpy as np
from win32comext.adsi.demos.scp import verbose

from run_EKF import run_EKF
from thymio import Thymio
from ComputerVision import ComputerVision
from ExtendedKF import KalmanFilterExtended
from global_navigation import Navigation


def get_frame_with_vectors(vision, frame):
    frame_with_markers, marker_ids, rvecs, tvecs, aruco_diagonal_pixels = vision.detect_and_estimate_pose(frame)

    # Will contain ID, x, y, angle of each detected marker
    markers_data = []

    if marker_ids is not None:
        frame_with_vectors, markers_data = vision.process_marker_pose(frame_with_markers, marker_ids, rvecs, tvecs)
    else:
        frame_with_vectors = frame.copy()

    # Conversion from pixels to mm
    conversion_factor = aruco_diagonal_pixels / (math.sqrt(2) * 45)

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


def convert_2_pixels(coordinates, conversion_factor):
    return int(round(coordinates[0] * conversion_factor)), int(round(coordinates[1] * conversion_factor))


def draw_obstacles(frame, navigation, conversion_factor):
    for shape in navigation.obstacles:
        for x, y in shape:
            x, y = convert_2_pixels((x, y), conversion_factor)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for shape in navigation.get_extended_obstacles():
        for x, y in shape:
            x, y = convert_2_pixels((x, y), conversion_factor)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


def draw_path(frame, path, conversion_factor):
    prev_point = path[0]

    x_prev, y_prev = convert_2_pixels((prev_point.x, prev_point.y), conversion_factor)
    # First point
    cv2.circle(frame, (int(prev_point.x * conversion_factor), int(prev_point.y * conversion_factor)), 5, (0, 0, 255), -1)

    for point in path[1:]:
        x, y = convert_2_pixels((point.x, point.y), conversion_factor)

        # Line from point to point
        cv2.line(frame, (x_prev, y_prev), (x, y), (0, 0, 255), 2)
        # Next point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        prev_point = point
        x_prev, y_prev = x, y


if __name__ == "__main__":
    # INITIALIZATION
    verbose = False
    NUMBER_OF_OBSTACLES = 3


    # COMPUTER VISION
    vision = ComputerVision(camera_index=1, robot_id=4, goal_id=8)
    corners = []

    ret, frame = vision.cam.read()
    # While loop to avoid noisy detections when camera starts
    while True:
        if not ret:
            print("Failed to capture frame. Exiting.")

        frame_masked = vision.apply_color_mask(frame,  mask_color='r')  # We apply a red mask to detect only red obstacles
        frame_with_vectors, markers_data, marker_ids, conversion_factor = get_frame_with_vectors(vision, frame)
        edges = vision.detect_edges(frame_masked)  # We detect the edges on the masked frame using Canny
        corners = vision.get_corners_and_shape_edges(edges)  # We obtain shapes and corners by approxPolyDP

        ret, frame = vision.cam.read()

        # If all the obstacles are detected, we can break the loop
        if len(corners) == NUMBER_OF_OBSTACLES:
            break

    # Conversion from pixels to mm
    corners_mm = [np.array(shape) / conversion_factor for shape in corners]


    # THYMIO
    thymio = Thymio()
    thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)
    goal_pos = get_goal_position(markers_data)
    if verbose:
        print("Goal position: ", goal_pos)
        print("Thymio position: ", thymio_pos_x, thymio_pos_y)
        print("Conversion factor: ", conversion_factor)


    # GLOBAL NAVIGATION
    global_goal = deepcopy(goal_pos)
    global_obstacles = deepcopy(corners_mm)
    navigation = Navigation(global_obstacles, [thymio_pos_x, thymio_pos_y], global_goal)
    global_path = navigation.get_shortest_path()
    check_num = 0
    drawing_path = deepcopy(global_path)

    # Thymio goal setting
    goal_pos[0] = -goal_pos[0]
    thymio.set_goal(goal_pos)

    # create and initialize the Kalman filter
    u = thymio.get_wheels_speed()
    ekf = KalmanFilterExtended(np.array([thymio_pos_x * -1, thymio_pos_y, thymio_theta]), u)
    camera = True

    # MAIN LOOP
    while not thymio.is_on_goal:
        # LOCAL NAVIGATION
        if not thymio.is_kidnapped and np.any(thymio.get_horizontal_sensors() > thymio.OBSTACLE_THRESHOLD):
            thymio.local_navigation()

        ### Detection and position of the markers (ROBOT and GOAL) ###
        ret, frame = vision.cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        ### Detection of red edges and red corners ###
        frame_masked = vision.apply_color_mask(frame, mask_color='r')   # We apply a red mask to detect only red obstacles
        edges = vision.detect_edges(frame_masked)                       # We detect the edges on the masked frame using Canny
        corners = vision.get_corners_and_shape_edges(edges)             # We obtain shapes and corners by approxPolyDP

        frame_with_vectors, markers_data, marker_ids, _ = get_frame_with_vectors(vision, frame)
        frame_aruco_and_corners = frame_with_vectors.copy()

        # Draw obstacles and path
        draw_obstacles(frame_aruco_and_corners, navigation, conversion_factor)
        draw_path(frame_aruco_and_corners, drawing_path, conversion_factor)


        ### Detection if camera is covered ###
        # We check if self.ARUCO_ROBOT_ID is in the detected markers
        camera_covered = vision.is_camera_covered(marker_ids)
        if camera_covered:
            goal_pos = get_goal_position(markers_data)
            if goal_pos is None:
                print("Camera is covered!")
                camera = False
            else:
                print("Thymio is being kidnapped")
                thymio.kidnap()
                continue


        # THYMIO POSITION
        thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)


        # KALMAN FILTER
        u = thymio.get_wheels_speed()
        theta = ((thymio_theta + 180) % 360) * np.pi / 180

        measured_state = run_EKF(ekf, thymio_pos_x * -1, thymio_pos_y, theta, u, cam=camera)

        x = int(round(measured_state[0] * -1 * conversion_factor))
        y = int(round(measured_state[1] * conversion_factor))
        cv2.circle(frame_aruco_and_corners, (x, y), 5, (255, 255, 0), -1)

        cv2.imshow("Main frame", frame_aruco_and_corners)

        # ACCOUNT FOR KIDNAPPING
        if thymio.is_kidnapped:
            thymio.recover()
            navigation = Navigation(global_obstacles, [thymio_pos_x, thymio_pos_y], global_goal)
            global_path = navigation.get_shortest_path()
            drawing_path = deepcopy(global_path)
            check_num = 0

            u = thymio.get_wheels_speed()
            ekf = KalmanFilterExtended(np.array([thymio_pos_x * -1, thymio_pos_y, thymio_theta]), u)
            camera = True

        thymio_theta = (thymio_theta + 180) % 360
        thymio_pos_x = -thymio_pos_x
        thymio.set_position(np.array([thymio_pos_x, thymio_pos_y]))
        thymio.set_orientation(thymio_theta * np.pi / 180)

        # MOVEMENT
        target = global_path[0]
        if thymio.move_to_point(np.array([target.x * -1, target.y])):
            print(f"Reached checkpoint {check_num}")
            check_num += 1
            global_path.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            thymio.stop()
            break

    vision.release()
    thymio.stop()
    thymio.__del__()
    print("Goal reached")