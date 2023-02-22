import os

import numpy as np
from matplotlib import pyplot as plt

from uav_env_contrast import Environment_2D
from jueru.Agent_set import DDPG_agent
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--max_episode_steps', type=int, default=500)
# parser.add_argument('--pursuer_isTotal_range', type=bool, default=False)
# parser.add_argument('--pursuer_lidar_lines', type=int, default=11)
# parser.add_argument('--pursuer_radius_thr', type=float, default=20.0)
# parser.add_argument('--evader_isTotal_range', type=bool, default=True)
# parser.add_argument('--evader_lidar_lines', type=int, default=10)
# parser.add_argument('--evader_radius_thr', type=float, default=10.0)
# parser.add_argument('--evader_move', type=bool, default=True)
# parser.add_argument('--velocity_bound', type=float, default=0.5)
# parser.add_argument('--angle_bound', type=float, default=np.pi / 6)
# parser.add_argument('--max_velocity', type=float, default=2.0)
# parser.add_argument('--evader_xy_speed', type=float, default=1.0)

# args = parser.parse_args()

def eval(env, agent , num_episode):
    obs = env.reset()
    target_count = 0
    count_episode = 0
    episode_reward = 0
    list_episode_reward = []
    success_num = 0
    while count_episode < num_episode:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        episode_reward += reward
        # env.render()
        if env.isSuccessful or done:
            if env.isSuccessful:
                success_num+=1
            obs = env.reset()
            count_episode += 1
            list_episode_reward.append(episode_reward.copy())
            episode_reward = 0
    success_rate = success_num / num_episode
    average_reward = sum(list_episode_reward) / len(list_episode_reward)
    #self.writer.add_scalar('eval_average_reward', average_reward, global_step=step)
    print('success', success_rate)
    return success_rate

agent = DDPG_agent.load('DDPG_model_address')
disturbance_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ddpg_success_rate_list = []
for disturb in disturbance_list:
    env = Environment_2D(disturbance=disturb)

    # obs = env.reset(speed_random=True)
    # # env.render(mode='rgb_array')
    # target_count = 0
    # step = 0

    success_rate = eval(env, agent, 1000)
    ddpg_success_rate_list.append(success_rate)
    env.close()

agent = DDPG_agent.load('Trick_DDPG_model_address')
trick_ddpg_success_rate_list = []
for disturb in disturbance_list:
    env = Environment_2D(disturbance=disturb)

    # obs = env.reset(speed_random=True)
    # # env.render(mode='rgb_array')
    # target_count = 0
    # step = 0

    success_rate = eval(env, agent, 1000)
    trick_ddpg_success_rate_list.append(success_rate)
    env.close()
plt.clf()
plt.plot(disturbance_list, ddpg_success_rate_list, color='b', label = 'ddpg')

plt.plot(disturbance_list, trick_ddpg_success_rate_list, color='g', label='trick_ddpg')

plt.legend(loc='best')

plt.savefig('robust_line.png', dpi=1000)
plt.clf()



