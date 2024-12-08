import numpy as np
from matplotlib import pyplot as plt
from tdmclient import ClientAsync, aw
import time


class Thymio():
    SENSORS_HORIZONTAL = "prox.horizontal"
    MOTOR_LEFT = "motor.left.target"
    MOTOR_RIGHT = "motor.right.target"

    SPEED_LEFT = "motor.left.speed"
    SPEED_RIGHT = "motor.right.speed"

    GOAL_THRESHOLD = 20
    OBSTACLE_THRESHOLD = 1800
    ANGLE_THRESHOLD = 0.01

    SCALE = 0.03
    SPEED = 70

    W = np.array([[2, 1, -4, -1, -2], [-2, -1, -2, 1, 2]]) * SCALE

    SHARP_TURN_THRESHOLD = np.pi / 6

    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())

        self.goal = None
        self.position = None
        self.orientation = None
        self.is_on_goal = False
        self.is_kidnapped = False

    def __del__(self):
        aw(self.node.unlock())

    def set_motors(self, left_motor: int, right_motor: int, verbose: bool = False):
        """
        Sets the motor speeds of the Thymio

        :param l_speed: left motor speed
        :param r_speed: right motor speed
        :param verbose: whether to print status messages or not
        """

        if verbose:
            print("\t\t Setting speed : ", left_motor, right_motor)

        # aw(self.client.wait_for_status(self.client.NODE_STATUS_READY))
        aw(self.node.set_variables({
            self.MOTOR_LEFT: [left_motor],
            self.MOTOR_RIGHT: [right_motor]
        }))

    def get_horizontal_sensors(self, verbose: bool = False):
        """
        Returns the horizontal proximity sensors

        :param verbose: whether to print status messages or not
        """

        aw(self.client.wait_for_status(self.client.NODE_STATUS_READY))
        aw(self.node.wait_for_variables({self.SENSORS_HORIZONTAL}))

        values = self.node.var[self.SENSORS_HORIZONTAL]

        if verbose:
            print("\t\t Sensor values : ", values)

        return np.array(values)[:5]

    def get_wheels_speed(self, verbose: bool = False):
        """
        Returns the speed of the wheels

        :param verbose: whether to print status messages or not
        """

        aw(self.client.wait_for_status(self.client.NODE_STATUS_READY))
        aw(self.node.wait_for_variables({self.SPEED_LEFT, self.SPEED_RIGHT}))

        left_speed = self.node.var[self.SPEED_LEFT]
        right_speed = self.node.var[self.SPEED_RIGHT]

        if verbose:
            print("\t\t Wheel speeds : ", left_speed, right_speed)

        return np.array([left_speed[0], right_speed[0]])

    def local_navigation(self):
        """
        Local navigation until no obstacles are detected

        :return: True if no obstacles are detected
        """
        while np.any(self.get_horizontal_sensors() > self.OBSTACLE_THRESHOLD):
            motor_values = self.W @ self.get_horizontal_sensors().T
            left_motor = int(motor_values[0])
            right_motor = int(motor_values[1])

            self.set_motors(left_motor, right_motor)

        self.set_motors(self.SPEED, self.SPEED)
        time.sleep(1.5)

    def stop(self):
        """
        Stops the Thymio
        """
        self.set_motors(0, 0)

    def kidnap(self):
        """
        Kidnaps the Thymio
        """
        self.is_kidnapped = True
        self.stop()

    def recover(self):
        """
        Recovers the Thymio from kidnapping
        """
        self.is_kidnapped = False

    def set_goal(self, goal):
        """
        Sets the goal for the Thymio
        :param goal: goal position
        """
        self.goal = goal

    def set_position(self, position: np.ndarray):
        """
        Sets the position of the Thymio
        :param position: position of the Thymio
        """
        self.position = position

    def set_orientation(self, orientation):
        """
        Sets the orientation of the Thymio
        :param orientation: orientation of the Thymio
        """
        self.orientation = orientation

    def move_to_point(self, target: np.ndarray, verbose: bool = False):
        """
        Turn and move the Thymio towards the goal depending on the position and the next target
        :param target: next target position
        :param verbose: whether to print status messages or not
        """
        path = target - self.position

        dst = np.sqrt(np.sum(np.square(path)))

        if dst <= self.GOAL_THRESHOLD:
            if np.sqrt(np.sum(np.square(self.position - self.goal))) <= self.GOAL_THRESHOLD:
                self.is_on_goal = True

            return True

        angle = self.orientation - np.arctan2(path[1], path[0])

        if angle > np.pi:
            angle = angle - 2 * np.pi

        if verbose:
            print("Distance: ", dst)
            print("Orientation: ", self.orientation)
            print("Angle: ", angle)
            print("Position: ", self.position)
            print("Target: ", target)

        if angle > self.ANGLE_THRESHOLD:
            if angle > self.SHARP_TURN_THRESHOLD:
                self.set_motors(self.SPEED, -self.SPEED)
                time.sleep(0.5)
            else:
                self.set_motors(self.SPEED, -self.SPEED)
        elif angle < -self.ANGLE_THRESHOLD:
            if angle < -self.SHARP_TURN_THRESHOLD:
                self.set_motors(-self.SPEED, self.SPEED)
                time.sleep(0.5)
            else:
                self.set_motors(-self.SPEED, self.SPEED)

        self.set_motors(self.SPEED, self.SPEED)

        return False

    def plot_direction(self, goal):
        """
        Plot the direction of the Thymio
        """
        plt.plot(self.position[0], self.position[1], 'ro', label='Thymio')
        plt.plot(goal[0], goal[1], 'bo', label='Goal')
        plt.quiver(self.position[0], self.position[1], np.cos(self.orientation), np.sin(self.orientation), minshaft=2, color='r', label='Orientation')
        plt.quiver(self.position[0], self.position[1], goal[0] - self.position[0], goal[1] - self.position[1], minshaft=2, color='b', label='Direction')
        max_x = max(abs(self.position[0]), abs(goal[0])) + 0.5
        max_y = max(abs(self.position[1]), abs(goal[1])) + 0.5
        plt.xlim(-max_x, max_x)
        plt.ylim(-max_y, max_y)
        plt.legend()
        plt.grid()
        plt.show()
