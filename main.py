import cv2
from thymio import Thymio
from ComputerVision import ComputerVision


if __name__ == "__main__":

    vision = ComputerVision(1)

    # create and initialize the Kalman filter


    # create and initialize the Thymio
    thymio = Thymio()
    thymio.set_position(...)
    thymio.set_orientation(...)

    while not thymio.is_on_goal:
        # local navigation
        if ...:
            thymio.local_navigation()



        ### Detection and position of the markers (ROBOT and GOAL) ###
        ret, frame = vision.cam.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # rvecs, tvecs are the rotation and translation vectors of the detected markers
        frame_with_markers, marker_ids, rvecs, tvecs = vision.detect_and_estimate_pose(frame)


        markers_data = []
        # Will contain ID, x, y, angle of each detected marker
        
        if marker_ids is not None:
            frame_with_vectors, markers_data = vision.process_marker_pose(frame_with_markers, marker_ids, rvecs, tvecs)
        else:
            frame_with_vectors = frame.copy()



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