import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pyvisgraph as vg

THYMIO_RADIUS = 1

# Navigation class
class Navigation:
    def __init__(self):
        self.obstacles = []
        self.obstacles_count = 0
        self.extended_obstacles = []
        self.source = []
        self.goal = []
        self.path = []

    def set_obstacles_source_goal(self, obstacles, robot, goal):
        """
        :param obstacles:  A list of all the obstacles on the map. An obstacle is a list of 
                            vertices, ordered CCW.
        :param robot:      The position of the robot.
        :param goal:       The position of the goal.
        """
        self.obstacles = obstacles
        self.obstacles_count = len(obstacles)
        self.source = robot
        self.goal = goal


    @staticmethod
    def augment_obtacles(self):
        """
        Augments the countour of the obstacles w.r.t. the radius of the robot.
        """
        for polygon in self.obstacles:
            print(polygon)
            extended_polygon = []

            count = len(polygon)
            for i in range(count):
                vertex = polygon[i]
                prev = polygon[i-1]
                next = (polygon[i+1] if i < count-1 else polygon[0])

                edge1 = np.subtract(vertex,prev)
                edge2 = np.subtract(next,vertex)

                dir1 = edge1 / np.linalg.norm(edge1)
                dir2 = -edge2 / np.linalg.norm(edge2)

                extended_polygon.append(vertex + THYMIO_RADIUS * dir2)
                extended_polygon.append(vertex + THYMIO_RADIUS * dir1)

            self.extended_obstacles.append(extended_polygon)

        print(self.extended_obstacles)


    def get_shortest_path(self):
        """
        Computes the shortest path from source (robot) to goal, while avoiding the obstacles.

        :returns:
            A list of the coordinates of each node from robot position to goal.
        """
        graph = vg.VisGraph()
        polygons = []
        for obstacle in self.extended_obstacles:
            polygon = []
            for point in obstacle:
                polygon.append(vg.Point(point[0], point[1]))

            polygons.append(polygon)
                
        graph.build(polygons)
        path = graph.shortest_path(vg.Point(self.source[0], self.source[1]), vg.Point(self.goal[0], self.goal[1]))

        for point in path:
            self.path.append(point)

        return self.path

# obstacles = [[[0,1],[3,1],[1.5,4]],
#              [[4,4],[7,4],[5.5,8]]]
# nav = Navigation(obstacles,[1.5,0],[4,6])
# nav.augment_obtacles()
# path = nav.get_shortest_path()
# print(path)
