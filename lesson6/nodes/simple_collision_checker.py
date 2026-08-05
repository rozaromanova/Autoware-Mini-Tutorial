#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

# Collision point categories: 0 = none, 1 = goal, 2 = traffic light, 3 = static obstacle, 4 = moving obstacle
DTYPE = np.dtype([
    ('x', np.float32),          # position
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),         # velocity
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),   # safety distance before collision point
    ('deceleration_limit', np.float32), # max allowed deceleration (np.inf = no limit)
    ('category', np.int32)
])


class SimpleCollisionChecker:

    def __init__(self):

        # Parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        # TODO 8 (lesson 7): add braking_safety_distance_stopline parameter,
        #                    load the lanelet2 map and extract the stop lines with traffic lights

        # Variables
        self.detected_objects = None
        self.goal_point = None
        # TODO 8 (lesson 7): add stopline_statuses dict

        # Lock for thread safety
        self.lock = threading.Lock()

        # Publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=None, tcp_nodelay=True)
        # TODO 8 (lesson 7): add traffic_light_status subscriber

        rospy.loginfo("%s - initialized", rospy.get_name())

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def global_path_callback(self, msg):
        if len(msg.waypoints) > 0:
            self.goal_point = msg.waypoints[-1].position
        else:
            self.goal_point = None

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
            goal_point = self.goal_point

        collision_points = np.array([], dtype=DTYPE)

        if len(msg.waypoints) == 0:
            collision_points_msg = PointCloud2()
            collision_points_msg.header = msg.header
            self.collision_points_pub.publish(collision_points_msg)
            return

        # Create a Shapely LineString from the local path waypoints
        local_path_linestring = shapely.geometry.LineString([(wp.position.x, wp.position.y) for wp in msg.waypoints])
        local_path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
        shapely.prepare(local_path_buffer)

        if detected_objects is not None and len(detected_objects) > 0:
            for obj in detected_objects:
                if not getattr(obj, "convex_hull", None):
                    continue

                hull_points = []
                for i in range(0, len(obj.convex_hull), 3):
                    if i + 1 >= len(obj.convex_hull):
                        break
                    hull_points.append((obj.convex_hull[i], obj.convex_hull[i + 1]))

                if len(hull_points) < 3:
                    continue

                object_polygon = shapely.Polygon(hull_points)

                if local_path_buffer.intersects(object_polygon):
                    intersection_geometry = local_path_buffer.intersection(object_polygon)
                    intersection_points = shapely.get_coordinates(intersection_geometry)

                    for x, y in intersection_points:
                        speed = math.sqrt(obj.velocity.x ** 2 + obj.velocity.y ** 2 + obj.velocity.z ** 2)
                        category = 4 if speed > self.stopped_speed_limit else 3
                        collision_points = np.append(
                            collision_points,
                            np.array(
                                [(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                                  self.braking_safety_distance_obstacle, np.inf, category)],
                                dtype=DTYPE,
                            ),
                        )

        if goal_point is not None:
            goal_point_shapely = shapely.Point(goal_point.x, goal_point.y)
            if local_path_buffer.intersects(goal_point_shapely.buffer(0.1)):
                collision_points = np.append(
                    collision_points,
                    np.array(
                        [(goal_point.x, goal_point.y, 0.0, 0.0, 0.0, 0.0,
                          self.braking_safety_distance_goal, np.inf, 1)],
                        dtype=DTYPE,
                    ),
                )

        # TODO 9 (lesson 7): add stop line collision points for red traffic lights

        # Publish the collision points (an empty array means no collision points on the path)
        if len(collision_points) > 0:
            collision_points_msg = msgify(PointCloud2, collision_points)
        else:
            collision_points_msg = PointCloud2()
        collision_points_msg.header = msg.header
        self.collision_points_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('simple_collision_checker', log_level=rospy.INFO)
    node = SimpleCollisionChecker()
    node.run()
