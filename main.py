import math
import os
from copy import deepcopy
import sys

import cv2
import numpy as np

from run_EKF import run_EKF
from thymio import Thymio
from ComputerVision import ComputerVision
from ExtendedKF import KalmanFilterExtended
from global_navigation import Navigation


def get_frame_with_vectors(vision, frame):
    """
    Get the frame with the markers and vectors
    :param vision: vision object
    :param frame: frame to process
    """
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
    """
    Get the position and orientation of the Thymio
    :param markers_data: data of the detected markers
    """
    thymio_pos_x, thymio_pos_y, thymio_theta = None, None, None

    for marker in markers_data:
        if marker[0] == vision.ARUCO_ROBOT_ID:
            thymio_pos_x, thymio_pos_y = marker[1], marker[2]
            thymio_theta = marker[3]

    return thymio_pos_x, thymio_pos_y, thymio_theta


def get_goal_position(markers_data):
    """
    Get the position of the goal
    :param markers_data: data of the detected markers
    """
    goal_pos = None

    for marker in markers_data:
        if marker[0] == vision.ARUCO_GOAL_ID:
            goal_pos = [marker[1], marker[2]]
            break

    return goal_pos


def convert_2_pixels(coordinates, conversion_factor):
    """
    Convert coordinates from mm to pixels
    :param coordinates: coordinates in mm
    :param conversion_factor: conversion factor from pixels to mm
    """
    return int(round(coordinates[0] * conversion_factor)), int(round(coordinates[1] * conversion_factor))


def draw_obstacles(frame, navigation, conversion_factor):
    """
    Draw the obstacles on the frame
    :param frame: frame to draw on
    :param navigation: navigation object
    """
    for shape in navigation.obstacles:
        for x, y in shape:
            x, y = convert_2_pixels((x, y), conversion_factor)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for shape in navigation.get_extended_obstacles():
        for x, y in shape:
            x, y = convert_2_pixels((x, y), conversion_factor)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


def draw_path(frame, path, conversion_factor):
    """
    Draw the path on the frame
    :param frame: frame to draw on
    :param path: path to draw
    :param conversion_factor: conversion factor from pixels to mm
    """
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


def draw_ekf(frame, measured_state, conversion_factor, P, theta):
    """
    Draw the ellipse representing the confidence of the Kalman filter
    :param frame: frame to draw on
    :param measured_state: state of the Kalman filter
    :param conversion_factor: conversion factor from pixels to mm
    :param P: covariance matrix
    """

    x = int(round(measured_state[0] * -1 * conversion_factor))
    y = int(round(measured_state[1] * conversion_factor))
    cv2.circle(frame_aruco_and_corners, (x, y), 5, (255, 255, 0), -1)

    mean = [x, y]


    # Extract eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(P)

    # Sort eigenvalues (largest first) and corresponding eigenvectors
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate the angle of the ellipse in degrees
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Calculate the axis lengths (scaled by confidence)
    chi_squared_val = np.sqrt(5.991)  # 95% confidence level for 2D
    amplification = 1
    major_axis = chi_squared_val * np.sqrt(eigenvalues[0]) * amplification
    minor_axis = chi_squared_val * np.sqrt(eigenvalues[1]) * amplification

    # Draw the ellipse
    center = (int(mean[0]), int(mean[1]))
    axes = (int(major_axis), int(minor_axis))
    cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 255), 2)


if __name__ == "__main__":
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    # INITIALIZATION
    verbose = False
    NUMBER_OF_OBSTACLES = int(sys.argv[1])

    if NUMBER_OF_OBSTACLES < 0:
        print("Please provide a number of obstacles greater than 0")
        os.close(0)


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


    # MARKERS
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
    thymio = Thymio()
    thymio.set_position(np.array([thymio_pos_x * -1, thymio_pos_y]))
    thymio.set_orientation(((thymio_theta + 180) % 360) * np.pi / 180)
    goal_pos[0] = -goal_pos[0]
    thymio.set_goal(goal_pos)

    # create and initialize the Kalman filter
    u = thymio.get_wheels_speed()
    ekf = KalmanFilterExtended(np.array([thymio_pos_x * -1, thymio_pos_y, ((thymio_theta + 180) % 360) * np.pi / 180]), u)
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
                print("Thymio is being kidnapped (marker not detected)")
                thymio.kidnap()
                continue
        else:
            camera = True


        # THYMIO POSITION
        if camera:
            thymio_pos_x, thymio_pos_y, thymio_theta = get_thymio_localisation(markers_data)
        else:
            thymio_pos_x, thymio_pos_y, thymio_theta = ekf.get_state()[:3]
            thymio_pos_x = -thymio_pos_x


        # KALMAN FILTER
        u = thymio.get_wheels_speed()
        theta = ((thymio_theta + 180) % 360) * np.pi / 180

        measured_state, kidnap = run_EKF(ekf, thymio_pos_x * -1, thymio_pos_y, theta, u, cam=camera)

        if not thymio.is_kidnapped and kidnap:
            thymio.kidnap()

        draw_ekf(frame_aruco_and_corners, measured_state, conversion_factor, ekf.get_cov()[:2, :2], thymio_theta)

        # out.write(frame_aruco_and_corners)
        cv2.imshow("Main frame", cv2.resize(frame_aruco_and_corners, (1920//2, 1080//2)))

        # ACCOUNT FOR KIDNAPPING
        if thymio.is_kidnapped:
            thymio.recover()
            navigation = Navigation(global_obstacles, [thymio_pos_x, thymio_pos_y], global_goal)
            global_path = navigation.get_shortest_path()
            drawing_path = deepcopy(global_path)
            check_num = 0

            u = thymio.get_wheels_speed()
            ekf = KalmanFilterExtended(np.array([thymio_pos_x * -1, thymio_pos_y, ((thymio_theta + 180) % 360) * np.pi / 180]), u)

        if camera:
            thymio_theta = ((thymio_theta + 180) % 360) * np.pi / 180
            thymio_pos_x = -thymio_pos_x
            thymio.set_position(np.array([thymio_pos_x, thymio_pos_y]))
            thymio.set_orientation(thymio_theta)
        else:
            thymio.set_position(np.array([measured_state[0], measured_state[1]]))
            thymio.set_orientation(measured_state[2])

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
    # out.release()
    thymio.stop()
    thymio.__del__()
    print("Goal reached")