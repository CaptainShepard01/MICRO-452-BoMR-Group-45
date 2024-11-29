import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


ARUCO_MARKER_SIDE_LENGTH = 45e-3  # Side length in meters
ARUCO_ROBOT_ID = 4
ARUCO_GOAL_ID = 10


def initialize_camera(camera_index=1):
    """
    Initialize the webcam.

    Param:
        camera_index: Index of the camera that we want to use (0 for laptop camera, 1 for webcam)
    
    Returns:
        cam
        frame_width
        frame_height
    """

    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise Exception("Camera not accessible")
    
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cam, frame_width, frame_height


def initialize_camera_parameters():
    """
    Initialize camera matrix and distortion coefficients. --- FROM FILE FOR THE REAL USE
    
    Returns:   
        mtx: camera matrix
        dst: distortion coefficients
    """

    # mtx = np.array([[1.47026991e+03, 0.00000000e+00, 9.78418512e+02],
    #                 [0.00000000e+00, 1.47625174e+03, 5.32097960e+02],
    #                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

    mtx = np.array([[1.47026991e+03, 0.00000000e+00, 0],
                    [0.00000000e+00, 1.47625174e+03, 0],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

    # dst = np.array([1.30382765e-01, -7.87531342e-01, -1.64764957e-03, 2.28727720e-05, 9.91067974e-01], dtype=np.float32) # np.zeros((5,), dtype=np.float32)


    # mtx = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]], dtype=np.float32)

    dst = np.zeros((5,), dtype=np.float32)
    
    return mtx, dst


def apply_color_mask(frame, mask_color='n'):
    """
    The function apply_color_mask takes an input frame, applies a color mask based on the specified
    color (red or blue), and returns the masked frame, or the original frame if no color is specified.

    Param:
        frame:      frame to be masked with the specified color.
        mask_color: color used for the mask. 'r' for red, 'b' for blue,
                    'n' to return the original frame (default)

    Returns:
        Frame with the specified color mask applied.
    """

    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    if mask_color == 'r':  # Red
        # Define both low-red and high-red ranges
        lower_red_1 = np.array([0, 100, 100])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 100, 100])
        upper_red_2 = np.array([180, 255, 255])

        # Combine the masks for both ranges
        mask_red_1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)
        mask_red_2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        # Apply morphological closing
        frame_masked = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    
    elif mask_color == 'b':  # Blue
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        mask_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)
        frame_masked = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    elif mask_color == 'n':
        frame_masked = frame

    return frame_masked


def detect_edges(frame):
    """
    Detect edges in the given frame using the Canny edge detector.

    :param frame: Input frame (BGR format expected).

    :return: Tuple containing the processed frame and its edges.
    """

    edges = cv2.Canny(frame, 50, 150)
    return edges


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
            
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
            
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
            
    return roll_x, pitch_y, yaw_z  # in radians


def detect_and_estimate_pose(original_frame, aruco_dict, aruco_params, mtx, dst):
    """
    Takes a frame as input, detects ArUco markers using the specified dictionary and parameters, 
    and estimates the pose of the detected markers using the camera matrix (`mtx`)
    
    :param original_frame: input frame/image where you want to detect ArUco markers and estimate
    their pose. 
    :param aruco_dict: dictionary containing the predefined markers that will be used for detection.
    These dictionaries define the marker size, border bits, and other properties necessary for marker
    detection and decoding.
    :param aruco_params: contains various parameters for the ArUco marker detection algorithm. These
    parameters can include values such as the detection mode, corner refinement method, minimum marker
    size, etc
    :param mtx: camera matrix which contains intrinsic parameters of the camera obtained by calibration.
    :param dst: distortion coefficients of the camera obtained by calibration.

    :return:
    1. `frame_with_markers`: The original frame with detected ArUco markers drawn on it.
    2. `marker_ids`: The IDs of the detected ArUco markers.
    3. `rvecs`: The rotation vectors estimated for each detected marker.
    4. `tvecs`: The translation vectors estimated for each detected marker.
    """

    frame_with_markers = original_frame.copy()

    corners, marker_ids, rejected = cv2.aruco.detectMarkers(frame_with_markers, aruco_dict, parameters=aruco_params)

    if marker_ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, marker_ids)

        # Estimate pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_SIDE_LENGTH,
            mtx,
            dst
        )
        return frame_with_markers, marker_ids, rvecs, tvecs
    return None, None, None, None


def process_marker_pose(original_frame, marker_ids, rvecs, tvecs, mtx, dst):
    """
    Process pose information for each detected marker and draw on a copy of the original frame.

    original_frame: The frame to process
    marker_ids, rvecs, tvecs: Detected marker details
    mtx, dst: Camera matrix and distortion coefficients

    Returns:
        frame_with_vectors: A new frame with the vectors drawn
    """
    frame_with_vectors = original_frame.copy()
    markers_data = []

    # Retourner une liste de ID, une liste de x,y,yaw et modifier l'image en dehors de la fonction

    for i, marker_id in enumerate(marker_ids):
        # Translation vector
        transform_translation_x = 1000*(tvecs[i][0][0]) # 1000* to have in millimeters.
        transform_translation_y = 1000*(tvecs[i][0][1])
        transform_translation_z = 1000*tvecs[i][0][2]

        # Rotation vector to quaternion
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()

        # Quaternion to Euler angles
        roll_x, pitch_y, yaw_z = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = -(math.degrees(yaw_z)-90)

        markers_data.extend([[marker_id[0], transform_translation_x, transform_translation_y, yaw_z]])

        cv2.drawFrameAxes(frame_with_vectors, mtx, dst, rvecs[i], tvecs[i], 0.05)

    return frame_with_vectors, markers_data


def reconstruct_image(original_frame, edges, robot_pos):
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    image_edge = cv2.addWeighted(original_frame, 1, edges_colored, 0.5, 0)
    reconstructed_image = cv2.addWeighted(robot_pos, 0.75, image_edge, 0.25, 0)

    return reconstructed_image


def get_corners_and_shape_edges(edges):
    """
    Detect contours in the edges image, approximate them using cv2.approxPolyDP,
    and return a list of corners for each detected shape.

    :param edges: Edge-detected image (binary format, result of cv2.Canny).
    :return: List of corners for each shape in the format [[[x1, y1], [x2, y2], ...], ...].
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store corners for each contour
    all_corners = []

    # Loop over each contour
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon for precision
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Extract corner points from the approximated polygon
        corners = [[int(point[0][0]), int(point[0][1])] for point in approx]
        all_corners.append(corners)

    return all_corners


def is_camera_covered(detected_markers):
    """
    Returns True (kidnapped) if the ARUCO_ROBOT_ID is not detected, False otherwise
    """
    if detected_markers is None:
        return True
    else:
        return ARUCO_ROBOT_ID not in detected_markers


def main():
    """
    Main function.
    """

    # Initialize camera parameters
    mtx, dst = initialize_camera_parameters()

    # Initialize ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # Start video capture
    cam, frame_width, frame_height = initialize_camera()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        ### ---------- DETECTION OF ARUCO MARKERS ---------- ###
        # Detect markers and estimate their pose
        frame_with_markers, marker_ids, rvecs, tvecs = detect_and_estimate_pose(frame, aruco_dict, aruco_params, mtx, dst)

        camera_covered = is_camera_covered(marker_ids)
        print(f"Robot not detected: {camera_covered} (ID robot = {ARUCO_ROBOT_ID})")


        # If markers are detected, process their pose and get a new frame
        markers_data = []
        if marker_ids is not None:
            frame_with_vectors, markers_data = process_marker_pose(frame_with_markers, marker_ids, rvecs, tvecs, mtx, dst)
        else:
            frame_with_vectors = frame.copy()


        ### ---------- DETECTION OF RED EDGES AND RED CORNERS ---------- ###
        # Detect red edges on the original frame
        frame_masked = apply_color_mask(frame, mask_color='r')
        red_edges = detect_edges(frame_masked)
        shapes_corners = get_corners_and_shape_edges(red_edges)

        # Draw corner points in red
        frame_aruco_and_corners = frame_with_vectors.copy()
        for shape in shapes_corners:
            for corner in shape:
                x, y = corner
                cv2.circle(frame_aruco_and_corners, (x, y), 5, (0, 255, 0), -1)


        cv2.imshow("Red corners + ARUCO vectors",  cv2.resize(frame_aruco_and_corners, (768, 432)))

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(markers_data)
            break


    # Release resources
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()