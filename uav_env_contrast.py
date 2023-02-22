# -*- coding:utf-8 -*-
"""
作者：509
日期：2021年03月14日
"""
import gym
import matplotlib
from gym import spaces
from stable_baselines3.common.env_checker import check_env

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib import animation
import csv
import imageio
from lidar import Lidar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_episode_steps', type=int, default=500)
parser.add_argument('--pursuer_isTotal_range', type=bool, default=False)
parser.add_argument('--pursuer_lidar_lines', type=int, default=11)
parser.add_argument('--pursuer_radius_thr', type=float, default=20.0)
parser.add_argument('--evader_isTotal_range', type=bool, default=True)
parser.add_argument('--evader_lidar_lines', type=int, default=10)
parser.add_argument('--evader_radius_thr', type=float, default=10.0)
parser.add_argument('--evader_move', type=bool, default=True)
parser.add_argument('--velocity_bound', type=float, default=0.5)
parser.add_argument('--angle_bound', type=float, default=np.pi / 6)
parser.add_argument('--max_velocity', type=float, default=2.0)
parser.add_argument('--evader_xy_speed', type=float, default=1.0)

args = parser.parse_args()

env_param_dict = {
    'width': 300,
    'height': 300,
    'obstacle_params': [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
                        (175, 75, 8), (175, 135, 8), (200, 100, 5), (225, 125, 5), (200, 85, 5),
                        (225, 170, 5), (200, 150, 5), (125, 100, 8), (225, 200, 5), (125, 175, 5),
                        (75, 100, 8), (100, 200, 5), (200, 150, 5), (100, 110, 8), (200, 170, 5), (160, 230, 8),
                        (60, 70, 10)]
}


class Environment_2D(gym.Env):
    """
	环境：二维，连续动作，连续状态
	目的：无人机追捕与逃避问题
	动作：角度，变化范围（0, 2*pi）；速度，变化范围（0，2），均为增量式
	状态：相对位置信息+环境障碍物信息
	"""

    def __init__(self, width=300, height=300,disturbance = 0):
        self.disturbance = disturbance
        self.param_set(width, height)
        self.create_evader()
        self.create_pursuer()
        self.create_obstacles()
        self.create_fig_ax()
        self.evader_lidar = Lidar(lidar_lines=args.evader_lidar_lines, isTotal_range=args.evader_isTotal_range,
                                  radius_thr=args.evader_radius_thr)
        self.pursuer_lidar = Lidar(lidar_lines=args.pursuer_lidar_lines, isTotal_range=args.pursuer_isTotal_range,
                                   radius_thr=args.pursuer_radius_thr)
        self.pursuer_effective_range = self.radius_thr - self.pursuer_radius
        self.K_DISTANCE = 0.05
        self.K_THETA = 0.2
        self.K_OBSTACLE = self.pursuer_effective_range / 10
        self.episode_steps = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)



    def param_set(self, width, height):
        """环境中相关超参数的初始化"""
        self.width = width
        self.height = height
        self.lidar_lines = args.pursuer_lidar_lines
        self.evader_speed = np.array([-1, -1])
        self.radius_thr = 20
        self.capture_distance = 15
        self.nActions = 2
        self.nStates_pos = 11
        self.nStates_obs = self.lidar_lines - 2
        self.nStates = self.nStates_pos + self.nStates_obs  # Remove the 0 degree and 180 degree rays
        self.cum_distance_history = 0
        self.angle_bound = args.angle_bound
        self.velocity_bound = args.velocity_bound
        # self.obstacle_params = [(25, 75, 8), (50, 125, 5), (75, 25, 5), (100, 100, 10), (125, 50, 8), (150, 175, 5),
        # 						(175, 50, 5), (150, 100, 5), (75, 175, 8), (175, 150, 5), (20, 20, 5)]  # (center_x, center_y, radius)
        # self.obstacle_params = [(25, 125, 8), (50, 75, 5), (75, 175, 5), (125, 150, 10), (125, 25, 8), (150, 100, 5),
        # 						(175, 75, 5), (150, 100, 5), (75, 50, 8),
        # 						(175, 150, 5), (75, 125, 5)]  # (center_x, center_y, radius)
        # self.obstacle_params = [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
        #                         (175, 75, 8), (200, 150, 5), (225, 125, 5), (200, 150, 5),
        #                         (125, 100, 8), (225, 200, 5), (125, 175, 5)]  # (center_x, center_y, radius)
        self.obstacle_params = [(75, 175, 8), (100, 125, 5), (125, 225, 5), (175, 200, 10),
                                (175, 75, 8), (175, 135, 8), (200, 100, 5), (225, 125, 5), (200, 85, 5),
                                (225, 170, 5), (200, 150, 5), (125, 100, 8), (225, 200, 5), (125, 175, 5),
                                (75, 100, 8), (100, 200, 5), (200, 150, 5), (100, 110, 8), (200, 170, 5), (160, 230, 8),
                                (60, 70, 10)]
        self.pursuer_growing_pos = [(100, 85), (100, 100), (100, 115)]
        self.growing_pos_index = 0
        self.obstacles = []
        self.isSuccessful = False

    def pdf(self, mu, sigma, x):
        """正态分布"""
        denominator = np.sqrt(2 * np.pi) * sigma
        return np.exp(-0.5 * np.square((x - mu) / sigma)) / denominator

    def sign(self, x, y):
        """符号函数"""
        if x > y:
            return 1
        else:
            return -1

    def distance_2d(self, x1, y1, x2, y2):
        """二维平面距离"""
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    def angel_diff_2d(self, x1, y1, x2, y2, alpha):
        """二维平面角度差"""
        direction_vec = np.array([np.cos(alpha), np.sin(alpha)])
        eva_pur_vec = np.array([x2 - x1, y2 - y1])
        angle_diff = np.arccos(
            np.sum(eva_pur_vec * direction_vec) / (np.linalg.norm(eva_pur_vec) + 1e-6))  # angel_diff belong to (0, pi)

        return angle_diff

    def get_true_action(self, action):  # [angle, velocity], belong to (-1, 1)
        """根据网络输出的动作转化为环境实际的动作"""
        action = np.clip(action, -1, 1)
        # angle
        action[0] = action[0] * self.angle_bound
        # velocity
        action[1] = action[1] * self.velocity_bound

        return action

    def pos_to_state(self, x1, y1, x2, y2, alpha):
        """根据智能体观测的信息组合成状态信息"""
        # --------------------------------------------------------------------------
        # 获取相对位置信息
        # --------------------------------------------------------------------------
        relative_distance_x = (x2 - x1) / self.width  # belong to (-1, 1)
        relative_distance_y = (y2 - y1) / self.height  # belong to (-1, 1)
        relative_distance = np.sqrt(
            (np.square(relative_distance_x) + np.square(relative_distance_y)) / 2)  # belong to (0, 1)
        relative_distance_norm = (relative_distance - 0.5) * 2
        relative_angle_norm = np.arctan2(y2 - y1, x2 - x1) / np.pi
        # 获取追踪者调整后的位置及速度信息
        pursuer_x_norm = ((self.pursuer_pos_x / self.width) - 0.5) * 2  # belong to (-1, 1)
        pursuer_y_norm = ((self.pursuer_pos_y / self.height) - 0.5) * 2  # belong to (-1, 1)
        pursuer_angle_norm = (self.pursuer_angle / np.pi) - 1  # belong to (-1, 1)
        pursuer_vel_norm = (2 * (self.pursuer_velocity / args.max_velocity)) - 1
        # 获取逃避者调整后的位置及速度信息
        evader_x_norm = ((self.evader_pos_x / self.width) - 0.5) * 2  # belong to (-1, 1)
        evader_y_norm = ((self.evader_pos_y / self.height) - 0.5) * 2  # belong to (-1, 1)
        evader_angle_norm = (self.evader_angle / np.pi) - 1  # belong to (-1, 1)
        evader_vel = np.array(self.evader_speed) / args.evader_xy_speed
        evader_vel_norm = (2 * np.sqrt(
            (np.square(evader_vel[0]) + np.square(evader_vel[1])) / 2)) - 1  # belong to (-1, 1)
        # 获取角度差信息
        angle_diff = self.angel_diff_2d(x1, y1, x2, y2, alpha)
        angle_diff_norm = ((angle_diff / np.pi) - 0.5) * 2  # 将角度截断在(-1, 1)之间，便于网络处理
        # 合成位置状态信息
        # s_pos = np.hstack((pursuer_x_norm, pursuer_y_norm, pursuer_angle_norm, relative_angle_norm,
        # 				   relative_distance_norm, evader_angle_norm, angle_diff_norm))
        s_pos = np.hstack((pursuer_x_norm, pursuer_y_norm, pursuer_angle_norm, pursuer_vel_norm,
                           evader_x_norm, evader_y_norm, evader_angle_norm, evader_vel_norm,
                           relative_angle_norm, relative_distance_norm, angle_diff_norm))
        # ---------------------------------------------------------------------------
        # 获取周围环境信息
        # ---------------------------------------------------------------------------
        self.scan_info = self.pursuer_lidar.scan_to_state(pos_x=self.pursuer_pos_x, pos_y=self.pursuer_pos_y,
                                                          angle=self.pursuer_angle,
                                                          obstacle_params=self.obstacle_params) - self.pursuer_radius
        self.scan_info = self.scan_info[1:-1]
        s_scan = self.scan_info / self.pursuer_effective_range
        s_scan_norm = (s_scan - 0.5) * 2  # 将环境信息截断在(-1, 1)之间，便于网络处理
        # 合成综合状态信息
        s = np.hstack((s_pos, s_scan_norm))

        return s

    def evader_reset(self, speed_random=False):
        """逃避者初始化"""
        # 圆心坐标
        self.evader_pos_x = 135
        self.evader_pos_y = 140
        if speed_random:
            self.evader_speed = np.random.uniform(-args.evader_xy_speed, args.evader_xy_speed, 2)
        self.evader_angle = np.arctan2(self.evader_speed[1], self.evader_speed[0])
        self.evader_angle = self.angle_normalize(self.evader_angle)

    def evader_random_reset(self, speed_random=False):
        """逃避者随机初始化"""
        self.evader_pos_x = np.random.choice([55, 245])
        self.evader_pos_y = np.random.uniform(60, 240)
        if speed_random:
            self.evader_speed = np.random.uniform(-args.evader_xy_speed, args.evader_xy_speed, 2)
        self.evader_angle = np.arctan2(self.evader_speed[1], self.evader_speed[0])
        self.evader_angle = self.angle_normalize(self.evader_angle)

    def pursuer_reset(self):
        """追踪者初始化"""
        # 圆心坐标
        self.pursuer_pos_x = 60
        self.pursuer_pos_y = 60
        # self.pursuer_angle = np.random.uniform(0, 2*np.pi)  # 起始位置角度在0-90度之间
        self.pursuer_angle = np.pi / 2  # 设定起始位置角度为90度
        self.pursuer_velocity = 1  # 设定初始化速度为1

    def pursuer_reset_v2(self, epoch, interval_step):
        """追踪者随机初始化"""
        # 圆心坐标
        if epoch % interval_step == 0:
            length = len(self.pursuer_growing_pos)
            self.growing_pos_index = np.random.randint(0, length)
        self.pursuer_pos_x = self.pursuer_growing_pos[self.growing_pos_index][0]
        self.pursuer_pos_y = self.pursuer_growing_pos[self.growing_pos_index][1]
        self.pursuer_angle = np.random.uniform(0, 2 * np.pi)
        self.pursuer_velocity = np.random.uniform(0, args.max_velocity)

    def pursuer_random_reset(self):
        """追踪者随机初始化"""
        self.pursuer_pos_x = 150
        if np.random.uniform() > 0.5:
            self.pursuer_pos_y = np.random.uniform(75, 125)
        else:
            self.pursuer_pos_y = np.random.uniform(175, 225)
        self.pursuer_angle = np.random.uniform(0, 2 * np.pi)
        self.pursuer_velocity = np.random.uniform(0, args.max_velocity)

    def create_evader(self):
        """创造逃避者"""
        self.evader_reset()
        self.evader_radius = 5
        self.evader = patches.Circle((self.evader_pos_x, self.evader_pos_y), radius=self.evader_radius, fc='g')

        # self.evader_traj = patches.Circle((self.evader_pos_x, self.evader_pos_y), radius=self.evader_radius, fc='g')

    def create_pursuer(self):
        """创造追踪者"""
        self.pursuer_reset()
        self.pursuer_radius = 5
        self.pursuer = patches.Circle((self.pursuer_pos_x, self.pursuer_pos_y), radius=self.pursuer_radius, fc='b')
        self.pursuer_field = patches.Circle((self.pursuer_pos_x, self.pursuer_pos_y),
                                            radius=self.radius_thr, fc=None, ec='k', alpha=0.3)

    def create_obstacles(self):
        """创造障碍物"""
        for obstacle_param in self.obstacle_params:
            self.obstacles.append(patches.Circle((obstacle_param[0], obstacle_param[1]),
                                                 radius=obstacle_param[2], fc='r'))

    def collision_detect(self):
        """碰撞检测"""
        for obstacle in self.obstacle_params:
            distance = self.distance_2d(self.pursuer_pos_x, self.pursuer_pos_y, obstacle[0], obstacle[1])
            if distance <= (self.pursuer_radius + obstacle[2]):
                return True

        return False

    def cross_border_detect(self):
        """跨界检测"""
        if (self.pursuer_pos_x < 0) or (self.pursuer_pos_x > self.width) \
                or (self.pursuer_pos_y < 0) or (self.pursuer_pos_y > self.height):
            return True
        else:
            return False

    def rep_field(self, scan_info):
        """人工势场"""
        distance_list = (1 / (scan_info + 1e-6)) - (1 / self.pursuer_effective_range)
        sin_angel_list = [np.sin((i * np.pi) / (args.pursuer_lidar_lines - 1)) for i in
                          range(1, args.pursuer_lidar_lines - 1)]
        cum_gravitation = np.sum(distance_list * np.array(sin_angel_list))

        return cum_gravitation

    def reward_mechanism(self, theta):
        """
        r_theta: Rewards and punishments brought by angle
        r_terminal: Rewards and punishments at the end of the round
        r_step: Rewards and punishments in step the environment
        r_distance: Rewards and punishments of distance change
        r_obstacles：Reward and punishment for obstacle avoidance
        r_total = r_theta + r_terminal + r_step + r_distance
        """
        self.episode_steps += 1
        # 终止状态标志
        done = False
        r_terminal = 0
        r_line = 0
        current_distance = self.distance_2d(self.pursuer_pos_x, self.pursuer_pos_y,
                                            self.evader_pos_x, self.evader_pos_y)
        isCollision = self.collision_detect()
        # 越界检测
        isCrossBorder = self.cross_border_detect()

        # r_terminal
        if self.episode_steps >= args.max_episode_steps:
            done = True
            if current_distance <= self.capture_distance:
                self.isSuccessful = True
            else:
                self.isSuccessful = False

        if isCollision or isCrossBorder:
            r_terminal = -10
            done = True
            self.isSuccessful = False

        # r_distance
        if (self.last_distance > self.capture_distance) and (current_distance > self.capture_distance):
            delta_dis = self.last_distance - current_distance
        elif (self.last_distance > self.capture_distance) and (current_distance <= self.capture_distance):
            delta_dis = self.last_distance - self.capture_distance + 5.0
        elif (self.last_distance <= self.capture_distance) and (current_distance > self.capture_distance):
            delta_dis = self.capture_distance - current_distance
        else:
            delta_dis = 5.0
        r_distance = self.K_DISTANCE * delta_dis
        self.last_distance = current_distance

        # r_theta
        r_theta = (self.K_THETA / np.pi) * ((np.pi / 2) - theta)
        # ------------------------------------------------------------------------------------------------
        # 计算障碍物奖励（法一）
        # ------------------------------------------------------------------------------------------------
        # r_obstacles_sign = self.sign(np.min(self.scan_info), np.min(self.last_scan_info))

        r_obstacles = -self.K_OBSTACLE * np.sum((1 / self.scan_info + 1e-6) - (1 / self.pursuer_effective_range))
        r_obstacles = np.clip(r_obstacles, -0.5, 0.)
        # ------------------------------------------------------------------------------------------------
        # 计算障碍物奖励（法二-人工势场法）
        # ------------------------------------------------------------------------------------------------
        # current_cum_gravitation = self.rep_field(self.scan_info)
        # r_obstacles_sign = self.sign(self.last_cum_gravitation, current_cum_gravitation)
        # r_obstacles = -self.K_OBSTACLE * current_cum_gravitation
        # r_obstacles = np.clip(r_obstacles, -0.5, 0.5)

        # r_line
        if args.pursuer_isTotal_range:
            if self.scan_info[0] < 10:
                r_line = ((self.scan_info[0] - 10) / 10) * self.K_THETA
        else:
            if self.scan_info[(len(self.scan_info) - 1) // 2] < 10:
                r_line = ((self.scan_info[(len(self.scan_info) - 1) // 2] - 10) / 10) * self.K_THETA

        # r_step
        # r_step = -(self.K_THETA/2)

        # r_total
        r_total = r_terminal + r_distance + r_theta + r_obstacles + r_line

        return r_total, done

    def angle_normalize(self, angle):
        """角度规则化"""
        # Adjust the angle to (0, 2*np.pi)
        angle %= 2 * np.pi

        return angle

    def velocity_normalize(self, velocity):
        """速度规则化"""
        # Adjust the velocity to (0, args.max_velocity)
        velocity = np.clip(velocity, 0, args.max_velocity)

        return velocity

    def reset(self, speed_random=True):
        """环境初始化"""
        # 重置迭代步数
        self.episode_steps = 0
        # 追踪者随机初始化
        self.pursuer_random_reset()

        # self.evader_reset(speed_random)
        self.evader_random_reset(speed_random=speed_random)
        self.last_distance = self.distance_2d(self.pursuer_pos_x, self.pursuer_pos_y,
                                              self.evader_pos_x, self.evader_pos_y)
        s = self.pos_to_state(self.pursuer_pos_x, self.pursuer_pos_y, self.evader_pos_x,
                              self.evader_pos_y, self.pursuer_angle)

        s = np.array(s).astype(np.float32)

        return s

    def step(self, action, isTrain=True):  # action -> [angel, velocity]
        """智能体执行动作，环境刷新"""
        # 将网络输出的动作映射到实际动作
        action = self.get_true_action(action)
        # 将角度规则化到（0，2*pi）
        self.pursuer_angle = self.angle_normalize(self.pursuer_angle + action[0])
        # 将速度规则化到（0，args.max_velocity）
        self.pursuer_velocity = self.velocity_normalize(self.pursuer_velocity + action[1])
        # 计算速度方向与两线方向的夹角
        theta = self.angel_diff_2d(self.pursuer_pos_x, self.pursuer_pos_y, self.evader_pos_x,
                                   self.evader_pos_y, self.pursuer_angle)
        # pursuer移动，计算位置
        pursuer_vel_x = self.pursuer_velocity * np.cos(self.pursuer_angle)
        pursuer_vel_y = self.pursuer_velocity * np.sin(self.pursuer_angle)
        self.pursuer_pos_x = self.pursuer_pos_x + pursuer_vel_x + self.disturbance * np.random.randn()
        self.pursuer_pos_y = self.pursuer_pos_y + pursuer_vel_y + self.disturbance * np.random.randn()
        # evader移动
        if args.evader_move:
            self.move_evader(gif=not isTrain)
        # 获取状态信息
        s_ = self.pos_to_state(self.pursuer_pos_x, self.pursuer_pos_y, self.evader_pos_x,
                               self.evader_pos_y, self.pursuer_angle)
        # 获取奖励信息和终止状态信息
        reward, done = self.reward_mechanism(theta)
        # 更新位置
        self.pursuer.set_center((self.pursuer_pos_x, self.pursuer_pos_y))
        self.pursuer_field.set_center((self.pursuer_pos_x, self.pursuer_pos_y))

        # if not isTrain:
        #     with open('./output_files/pursuer_trajectory.csv', mode='a') as csv_file:  # 根据文件复现目标运行轨迹
        #         writer = csv.writer(csv_file)
        #         writer.writerow([self.pursuer_pos_x, self.pursuer_pos_y])

        # info = '{}/{}'.format(self.episode_steps, args.max_episode_steps)
        info = {}
        s_ = np.array(s_).astype(np.float32)

        return s_, reward, done, info

    def evader_obs_avoid(self):
        """追踪者避障"""
        is_obs_exist = False
        evader_scan_info = self.evader_lidar.scan_to_state(pos_x=self.evader_pos_x, pos_y=self.evader_pos_y,
                                                           angle=self.evader_angle,
                                                           obstacle_params=self.obstacle_params)  # evader_scan_info: {ndarray: (12,)}
        cum_distance = np.sum(evader_scan_info)
        if np.asarray(evader_scan_info != self.evader_lidar.radius_thr).any():
            is_obs_exist = True

        return is_obs_exist, cum_distance

    def move_evader(self, gif=False):
        """追踪者移动"""
        is_obs_exist, cum_distance = self.evader_obs_avoid()
        # 判断是否改变方向
        if self.evader_pos_x < 50 or self.evader_pos_x > self.width - 50:
            self.evader_speed[0] = -self.evader_speed[0]
        if self.evader_pos_y < 50 or self.evader_pos_y > self.height - 50:
            self.evader_speed[1] = -self.evader_speed[1]
        if is_obs_exist:
            if cum_distance < self.cum_distance_history:
                if self.evader_speed[0] > 0:
                    self.evader_speed[0] = np.random.uniform(-args.evader_xy_speed, 0)
                else:
                    self.evader_speed[0] = np.random.uniform(0, args.evader_xy_speed)

                if self.evader_speed[1] > 0:
                    self.evader_speed[1] = np.random.uniform(-args.evader_xy_speed, 0)
                else:
                    self.evader_speed[1] = np.random.uniform(0, args.evader_xy_speed)
        self.cum_distance_history = cum_distance
        # 直线运动
        self.evader_pos_x = self.evader_pos_x + self.evader_speed[0]
        self.evader_pos_y = self.evader_pos_y + self.evader_speed[1]
        self.evader_angle = np.arctan2(self.evader_speed[1], self.evader_speed[0])
        self.evader_angle = self.angle_normalize(self.evader_angle)
        # 更新位置
        self.evader.set_center((self.evader_pos_x, self.evader_pos_y))

        if gif:
            with open('./output_files/evader_trajectory.csv', mode='a') as csv_file:  # 根据文件复现目标运行轨迹
                writer = csv.writer(csv_file)
                writer.writerow([self.evader_pos_x, self.evader_pos_y])

    def create_fig_ax(self):
        """创造绘图界面"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # plt.axis('equal')
        plt.grid()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks(np.arange(0, self.width + 1, 25))
        self.ax.set_yticks(np.arange(0, self.height + 1, 25))

        plt.ion()  # 开启interactive mode 成功的关键函数
        self.ax.add_patch(self.evader)
        self.ax.add_patch(self.pursuer)
        self.ax.add_patch(self.pursuer_field)
        for obstacle in self.obstacles:
            self.ax.add_patch(obstacle)

        plt.title('X-Y Plot')
        plt.xlabel('X')
        plt.ylabel('Y')

    def render(self, mode='human'):
        """绘图"""
        if mode == 'rgb_array':
            # self.ax.add_patch(self.evader_traj)
            plt.scatter(self.pursuer_pos_x,self.pursuer_pos_y,s = self.pursuer_radius/3,c ='b')
            plt.scatter(self.evader_pos_x, self.evader_pos_y, s=self.pursuer_radius / 3, c='g')
            plt.savefig('./output_images/image_sets/temp' + str(self.episode_steps) + '.png')
            image = imageio.imread('./output_images/image_sets/temp' + str(self.episode_steps) + '.png')
            return image

        elif mode == 'human':
            plt.show()
            plt.pause(0.0000001)

    # def move_gif(self):
    #     """获取动态图"""
    #
    #     def init():
    #         self.ax.add_patch(self.evader)
    #         self.ax.add_patch(self.pursuer)
    #         self.ax.add_patch(self.pursuer_field)
    #         for obstacle in self.obstacles:
    #             self.ax.add_patch(obstacle)
    #         return self.evader, self.pursuer,
    #
    #     def update(frame):
    #         evader = self.move_evader(gif=True)
    #         return evader,
    #
    #     anim = animation.FuncAnimation(self.fig, update, init_func=init,
    #                                    interval=500,
    #                                    blit=True)
    #     anim.save('../output_images/movie.gif', writer='pillow')
    #
    #     plt.show()
    #
    #     return


# def main():
#     env = Environment_2D()
#     while True:
#         env.render()
#         env.move_evader(gif=False)
#
#
# def test_step(env, angle):
#     s_, reward, done, _ = env.step(angle)
#     print('pursuer:', env.pursuer_pos_x, env.pursuer_pos_y, env.pursuer_angle)
#     print('evader: ', env.evader_pos_x, env.evader_pos_y)
#     print(s_)
#     print(s_.shape)
#     print(type(s_))
#     print('reward: ', reward)
#     print('done: ', done)
#     env.render()
#
#
# def test():
#     env = Environment_2D()
#     s = env.reset(speed_random=True)
#     env.pursuer_lidar.scan_to_state(pos_x=env.pursuer_pos_x, pos_y=env.pursuer_pos_y,
#                                     angle=env.pursuer_angle, obstacle_params=env.obstacle_params)
#     print('last distance: ', env.last_distance)
#     print('pursuer:', env.pursuer_pos_x, env.pursuer_pos_y, env.pursuer_angle)
#     print('evader: ', env.evader_pos_x, env.evader_pos_y)
#     print(s)
#     print(s.shape)
#     print(type(s))
#     print('last distance: ', env.last_distance)
#     env.render()
#     test_step(env, np.array([0, 0]))
#     test_step(env, np.array([0, 0]))
#     test_step(env, np.array([0, 0]))
#     test_step(env, np.array([0, 0]))


if (__name__ == '__main__'):
    env = Environment_2D()
    check_env(env)
    obs = env.reset()
    for i in range(1001):
        action = np.array([-0.5, 0.5])
        obs, reward, done, info = env.step(action)
        print('obs', obs.shape)
        env.render()
        if done:
            obs = env.reset()
            break
