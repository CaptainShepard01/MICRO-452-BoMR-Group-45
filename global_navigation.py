import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pyvisgraph as vg

THYMIO_RADIUS = 50 # mm

# Navigation class
class Navigation:
    def __init__(self, obstacles, robot, goal):
        self.obstacles = obstacles
        self.obstacles_count = len(obstacles)
        self.extended_obstacles = []
        self.source = robot
        self.goal = goal
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
            extended_polygon = []

            count = len(polygon)
            for i in range(count):
                vertex = polygon[i]
                prev = polygon[i-1]
                next = (polygon[i+1] if i < count-1 else polygon[0])

                edge1 = np.subtract(vertex, prev)
                edge2 = np.subtract(next,vertex)

                dir1 = np.array(edge1 / np.linalg.norm(edge1))
                dir2 = np.array(edge2 / np.linalg.norm(edge2))

                perp1 = -np.array([dir1[1], -dir1[0]])
                perp2 = -np.array([dir2[1], -dir2[0]])

                angle = np.arccos(np.clip(np.dot(-dir1, dir2), -1.0, 1.0))

                print(vertex)
                print(perp1, perp2)

                # v1 = vertex - THYMIO_RADIUS * dir2
                # v2 = vertex + THYMIO_RADIUS * dir1
                v1 = vertex + THYMIO_RADIUS * dir1 + THYMIO_RADIUS * perp1
                v2 = vertex - THYMIO_RADIUS * dir2 + THYMIO_RADIUS * perp2

                if angle <= np.pi / 2.0:
                    extended_polygon.append(v1)
                    extended_polygon.append(v2)
                else:
                    extended_polygon.append(v2)
                    extended_polygon.append(v1)

                # extended_polygon.append(0.5 * np.add(v1,v2))
                # extended_polygon.append(vertex - THYMIO_RADIUS * perp1)
                # extended_polygon.append(v1 + v2 - vertex)
                # extended_polygon.append(vertex - THYMIO_RADIUS * perp2)
                # extended_polygon.append(vertex + THYMIO_RADIUS * dir2)

            self.extended_obstacles.append(extended_polygon)


    def get_shortest_path(self):
        """
        Computes the shortest path from source (robot) to goal, while avoiding the obstacles.

        :returns:
            A list of the coordinates of each node from robot position to goal.
        """
        self.augment_obtacles(self)

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
            print(point)

        return self.path
    
    
    def get_extended_obstacles(self):
        return self.extended_obstacles




# obstacles = [[[147, 358], [83, 358], [81, 415], [146, 414]],
# [[547, 291], [428, 329], [549, 360]],
# [[402, 184], [213, 260], [299, 358]],
# [[77, 154], [78, 208], [141, 208], [140, 152]],
# [[297, 60], [300, 112], [359, 111], [356, 60]]]

# obstacles = [[[147, 358], [83, 358], [81, 415], [146, 414]]]
# obstacles = [[[547, 291], [428, 329], [549, 360]]]

# source = [50,430]
# goal = [550,50]

# nav = Navigation(obstacles,source,goal)
# path = nav.get_shortest_path()

# fig = plt.figure()
# plt.plot(source[0],source[1],'rs', label='source')
# plt.plot(goal[0],goal[1],'rx', label='goal')

# for shape in obstacles:
#     Xshape = []
#     Yshape = []
#     for x,y in shape:
#         Xshape.append(x)
#         Yshape.append(y)

#     Xshape.append(Xshape[0])
#     Yshape.append(Yshape[0])

#     plt.plot(Xshape,Yshape,'r-')

# X = []
# Y = []
# for shape in nav.get_extended_obstacles():
#     for x,y in shape:
#         X.append(x)
#         Y.append(y)
#         plt.plot(x,y,'bo')

# plt.plot(X,Y,'bo',label='padding')
    
# X = []
# Y = []
# for point in path:
#     X.append(point.x)
#     Y.append(point.y)

# plt.plot(X,Y,'g--', label='path')


# plt.legend()

# plt.show()