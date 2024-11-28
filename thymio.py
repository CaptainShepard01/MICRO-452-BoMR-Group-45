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

    LEDS_TOP = "leds.top"
    LEDS_BOTTOM_LEFT = "leds.bottom.left"
    LEDS_BOTTOM_RIGHT = "leds.bottom.right"
    LEDS_RC = "leds.rc"
    LEDS_TEMPERATURE = "leds.temperature"
    LEDS_PROX_H = "leds.prox.h"
    LEDS_PROX_V = "leds.prox.v"

    OBSTACLE_THRESHOLD = 1000
    SCALE = 0.01
    SPEED = 50

    goal = None
    position = None
    orientation = None
    is_on_goal = False

    K_RHO = 5
    K_ALPHA = 500
    K_BETA = -0.001
    WHEELBASE = 0.09

    ONETURN = 16 # ms for a full turn
    ANGLE_THRESHOLD = 0.01

    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())

        self.RAD_TURN = self.ONETURN / (2 * np.pi)

    def __del__(self):
        aw(self.node.unlock())

    def set_leds(self, leds: list, led_type: str):
        """
        Sets the LEDs of the Thymio

        :param leds: list of LED values
        :param led_type: type of LEDs to set
        """

        aw(self.node.set_variables({
            led_type: leds
        }))

    def reset_leds(self):
        """
        Resets the LEDs of the Thymio to 0
        """

        self.set_leds([0, 0, 0], self.LEDS_TOP)
        self.set_leds([0, 0, 0], self.LEDS_BOTTOM_LEFT)
        self.set_leds([0, 0, 0], self.LEDS_BOTTOM_RIGHT)
        self.set_leds([0], self.LEDS_RC)
        self.set_leds([0, 0], self.LEDS_TEMPERATURE)
        self.set_leds([0, 0, 0, 0, 0, 0, 0, 0], self.LEDS_PROX_H)
        self.set_leds([0, 0], self.LEDS_PROX_V)

    def set_motors(self, left_motor: int, right_motor: int, verbose: bool = False):
        """
        Sets the motor speeds of the Thymio

        :param l_speed: left motor speed
        :param r_speed: right motor speed
        :param verbose: whether to print status messages or not
        """

        if verbose:
            print("\t\t Setting speed : ", left_motor, right_motor)

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
            W = np.array([[2, 1, -1, -1, -2], [-2, -1, -1, 1, 2]]) * self.SCALE

            motor_values = W @ self.get_horizontal_sensors().T + self.SPEED
            left_motor = int(motor_values[0])
            right_motor = int(motor_values[1])

            self.set_motors(left_motor, right_motor)

    def stop(self):
        """
        Stops the Thymio
        """
        self.set_motors(0, 0)

    def set_goal(self, goal: np.ndarray):
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

    def set_orientation(self, orientation: np.ndarray):
        """
        Sets the orientation of the Thymio
        :param orientation: orientation of the Thymio
        """
        self.orientation = orientation

    def move_to_point(self, target: np.ndarray):
        """
        Turn and move the Thymio towards the goal depending on the position and the next target
        :param target: next target position
        """
        path = target - self.position
        angle = np.arctan2(path[0], path[1]) - self.orientation

        time_to_turn = abs(angle) * self.ONETURN / (2 * np.pi)

        if angle > self.ANGLE_THRESHOLD:
            self.set_motors(self.SPEED, -self.SPEED)
            time.sleep(time_to_turn)
        elif angle < -self.ANGLE_THRESHOLD:
            self.set_motors(-self.SPEED, self.SPEED)
            time.sleep(time_to_turn)

        self.set_motors(self.SPEED, self.SPEED)

    def move_to_point_astolfi(self, target: np.ndarray, verbose: bool = False):
        """
        Move the Thymio towards the goal depending on the position and the next target
        :param target: next target position
        """
        d = target - self.position
        rho = np.sqrt(np.sum(np.square(d)))
        if verbose:
            print("Orientation: ", self.orientation)
            print("Rho: ", rho)
            print("Dy: ", d[1])
            print("Dx: ", d[0])

        alpha = -self.orientation + np.arctan2(d[1], d[0])
        beta = -self.orientation - alpha

        v = self.K_RHO * rho
        w = self.WHEELBASE / 2 * (self.K_ALPHA * alpha + self.K_BETA * beta)

        left_motor = v - w
        right_motor = v + w

        if verbose:
            print("Left motor: ", left_motor)
            print("Right motor: ", right_motor)

        self.set_motors(int(np.floor(left_motor)), int(np.floor(right_motor)))

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
