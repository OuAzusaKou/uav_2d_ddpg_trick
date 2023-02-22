# -*- coding:utf-8 -*-
"""
作者：509
日期：2021年01月31日
"""
import numpy as np

class Lidar:
    def __init__(self, lidar_lines=8, isTotal_range=True, radius_thr=10):
        """
        Function description: Simulation of lidar, scanning the surrounding environment to obtain range information.
        :param lidar_lines: Number of rays emitted by lidar.
        :param isTotal_range: Scanning mode of lidar.
        :param radius_thr: Maximum scanning range of lidar.
        """
        self.lidar_lines = lidar_lines
        self.isTotal_range = isTotal_range
        self.radius_thr = radius_thr
        self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)

    def get_env_state(self, pos_x, pos_y, angle, obstacle_params):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.angle = angle
        self.obstacle_params = obstacle_params

    def distance_2d(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    def angle_normalize(self, angle):
        # Adjust the angle to (0, 2*np.pi)
        angle %= 2 * np.pi

        return angle

    def scan_distance(self, circle_center_distance, angle, obs):
        line_para = np.array([np.tan(angle), -1, self.pos_y - (self.pos_x * np.tan(angle))])
        dot_para = np.array([obs[0], obs[1], 1])
        dot_line_distance = np.abs(line_para @ dot_para) / np.sqrt(np.square(line_para[0]) + 1)
        distance_long = np.sqrt(np.square(circle_center_distance) - np.square(dot_line_distance))
        distance_short = np.sqrt(np.square(obs[2]) - np.square(dot_line_distance))
        distance_true = distance_long - distance_short

        return distance_true

    def distance_min(self, distance_trues):
        for index, distance_true in distance_trues:
            self.scan_info[index] = np.min([self.scan_info[index], distance_true])

    def get_scan_info(self, fov_obs, angles):
        # Get scan information
        line_angles = []
        distance_trues = []
        for obs in fov_obs:
            circle_center_distance = self.distance_2d(self.pos_x, self.pos_y, obs[0], obs[1])
            alpha_ = np.arctan2(obs[1] - self.pos_y, obs[0] - self.pos_x)
            alpha = self.angle_normalize(alpha_)
            beta_ = np.arcsin(obs[2] / circle_center_distance)
            beta = self.angle_normalize(beta_)
            if alpha >= beta and (alpha + beta) < 2 * np.pi:
                angle_low = alpha - beta
                angle_high = alpha + beta
                angle_low = self.angle_normalize(angle_low)
                angle_high = self.angle_normalize(angle_high)
                for index, angle in enumerate(angles):
                    if (angle >= angle_low) and (angle <= angle_high):
                        line_angles.append((index, angle))
                        distance_true = self.scan_distance(circle_center_distance, angle, obs)
                        distance_trues.append((index, distance_true))
                self.distance_min(distance_trues)
            else:
                angle_low = alpha - beta
                angle_high = alpha + beta
                angle_low = self.angle_normalize(angle_low)
                angle_high = self.angle_normalize(angle_high)
                for index, angle in enumerate(angles):
                    if ((angle >= angle_low) and (angle <= 2 * np.pi)) or ((angle >= 0) and (angle <= angle_high)):
                        line_angles.append((index, angle))
                        distance_true = self.scan_distance(circle_center_distance, angle, obs)
                        distance_trues.append((index, distance_true))
                self.distance_min(distance_trues)

    def scan_environment(self):
        # Giving UAVs the ability to scan the environment
        fov_obs = []  # Obstacle information in UAV field
        fov_bos_radius = []
        angles = []
        for obstacle in self.obstacle_params:

            distance = self.distance_2d(self.pos_x, self.pos_y, obstacle[0], obstacle[1]) - obstacle[2]
            if distance < self.radius_thr:
                fov_obs.append(obstacle)
                fov_bos_radius.append(obstacle[2])
        if len(fov_obs) != 0:
            if self.isTotal_range:  # Global range scanning
                if self.lidar_lines % 2:
                    self.lidar_lines += 1
                # scan_info: Information obtained by lidar scanning environment
                self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)
                single_angle = (2 * np.pi) / self.lidar_lines
                for i in range(0, self.lidar_lines):
                    angle_add = self.angle + (i * single_angle)
                    angle = self.angle_normalize(angle_add)
                    angles.append(angle)
                self.get_scan_info(fov_obs, angles)
            else:  # Semi global range scanning
                if self.lidar_lines % 2 == 0:
                    self.lidar_lines += 1
                # scan_info: Information obtained by lidar scanning environment
                self.scan_info = np.full((self.lidar_lines,), self.radius_thr, dtype=float)
                single_angle = np.pi / (self.lidar_lines - 1)
                for i in range(-(self.lidar_lines // 2), self.lidar_lines // 2 + 1):
                    angle_add = self.angle + (i * single_angle)
                    angle = self.angle_normalize(angle_add)
                    angles.append(angle)
                self.get_scan_info(fov_obs, angles)

    def scan_to_state(self, pos_x, pos_y, angle, obstacle_params):  # Interface API
        self.get_env_state(pos_x, pos_y, angle, obstacle_params)
        self.scan_environment()

        return self.scan_info