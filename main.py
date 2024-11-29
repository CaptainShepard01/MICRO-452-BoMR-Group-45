import cv2
from thymio import Thymio


if __name__ == "__main__":
    # initialize the camera


    # get position and angle of the thymio


    # create and initialize the Kalman filter


    # create and initialize the Thymio
    thymio = Thymio()
    thymio.set_position(...)
    thymio.set_orientation(...)

    while not thymio.is_on_goal:
        # local navigation
        if ...:
            thymio.local_navigation()

        # vision

        # Kalman filter

        # movement

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    thymio.stop()
    print("Goal reached")