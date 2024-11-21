import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_aruco_marker(marker_id, marker_size=200):
    """
    Generate an ArUco marker with the dictionary DICT_6X6_250

    Args:
        marker_id: ID of the marker to generate.
        side_pixels: Size of the marker image (in pixels).

    Returns:
        marker_image: Generated marker image.
    """

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    return marker_image


def detect_aruco_markers(image):
    """
    Detect ArUco markers in a given image using the dictionary DICT_6X6_250

    Args:
        image: Input image.

    Returns:
        corners: List of detected marker corners.
        ids: List of IDs of detected markers.
        image_with_markers: Image with detected markers outlined.
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    
    aruco_params = cv2.aruco.DetectorParameters()
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray_image)

    # Draw detected markers on the image
    image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

    return corners, ids, image_with_markers
