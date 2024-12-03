import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


class ComputerVision:
    def __init__(self, camera_index=1, aruco_marker_side_length=45e-3, robot_id=4, goal_id=8):
        self.ARUCO_MARKER_SIDE_LENGTH = aruco_marker_side_length
        self.ARUCO_ROBOT_ID = robot_id
        self.ARUCO_GOAL_ID = goal_id

        # Initialize camera parameters
        self.mtx, self.dst = self.initialize_camera_parameters()

        # Initialize ArUco dictionary and detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Initialize camera
        self.cam = self.initialize_camera(camera_index)
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def initialize_camera_parameters():
        """
        Initialize camera matrix and distortion coefficients. --- FROM FILE FOR THE REAL USE
        
        Returns:   
            mtx: camera matrix
            dst: distortion coefficients
        """

        mtx = np.array([[1.47026991e+03, 0.00000000e+00, 0],
                        [0.00000000e+00, 1.47625174e+03, 0],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
        dst = np.zeros((5,), dtype=np.float32)
    
        return mtx, dst


    @staticmethod
    def initialize_camera(camera_index):
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
        return cam


    @staticmethod
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

        if mask_color == 'r':  # Red mask
            lower_red_1 = np.array([0, 100, 100])
            upper_red_1 = np.array([10, 255, 255])
            lower_red_2 = np.array([170, 100, 100])
            upper_red_2 = np.array([180, 255, 255])
            
            mask_red_1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)
            mask_red_2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)
            mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

            frame_masked = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        elif mask_color == 'b':  # Blue mask
            lower_blue = np.array([110, 50, 50])
            upper_blue = np.array([130, 255, 255])

            mask_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)
            frame_masked = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        elif mask_color == 'n':
            frame_masked = frame
        
        return frame_masked


    @staticmethod
    def detect_edges(frame):
        """
        Detect edges in the given frame using the Canny edge detector.

        :param frame: Input frame (BGR format expected).

        :return: Tuple containing the processed frame and its edges.
        """
        return cv2.Canny(frame, 50, 150)


    @staticmethod
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
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z



    def detect_and_estimate_pose(self, frame):
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

        frame_with_markers = frame.copy()

        corners, marker_ids, _ = cv2.aruco.detectMarkers(frame_with_markers, self.aruco_dict, parameters=self.aruco_params)

        if marker_ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, marker_ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.ARUCO_MARKER_SIDE_LENGTH,
                self.mtx,
                self.dst)
            return frame_with_markers, marker_ids, rvecs, tvecs

        return None, None, None, None



    def process_marker_pose(self, frame, marker_ids, rvecs, tvecs):
        """
        Process pose information for each detected marker and draw on a copy of the original frame.

        original_frame: The frame to process
        marker_ids, rvecs, tvecs: Detected marker details
        mtx, dst: Camera matrix and distortion coefficients

        Returns:
            frame_with_vectors: A new frame with the vectors drawn
        """

        frame_with_vectors = frame.copy()
        markers_data = []

        for i, marker_id in enumerate(marker_ids):
            transform_translation_x = 1000*(tvecs[i][0][0]) # 1000* to have in millimeters.
            transform_translation_y = 1000*(tvecs[i][0][1])
            transform_translation_z = 1000*tvecs[i][0][2]

            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            roll_x, pitch_y, yaw_z = self.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = -(math.degrees(yaw_z)-90)

            markers_data.extend([[marker_id[0], transform_translation_x, transform_translation_y, yaw_z]])

            cv2.drawFrameAxes(frame_with_vectors, self.mtx, self.dst, rvecs[i], tvecs[i], 0.05)

        return frame_with_vectors, markers_data


    @staticmethod
    def get_corners_and_shape_edges(edges):
        """
        Detect contours in the edges image, approximate them using cv2.approxPolyDP,
        and return a list of corners for each detected shape.

        :param edges: Edge-detected image (binary format, result of cv2.Canny).
        :return: List of corners for each shape in the format [[[x1, y1], [x2, y2], ...], ...].
        """

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_corners = []

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)

            # Extract corner point from the approximated polygon
            corners = [[int(point[0][0]), int(point[0][1])] for point in approx]
            all_corners.append(corners)

        return all_corners



    def is_camera_covered(self, detected_markers):
        """
        Returns True (kidnapped) if the ARUCO_ROBOT_ID is not detected, False otherwise
        """

        if detected_markers is None:
            return True
        else:
            return self.ARUCO_ROBOT_ID not in detected_markers



    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()
