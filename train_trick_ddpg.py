import os
from copy import deepcopy

import numpy as np
import pytest
import torch as th
from gym import spaces, GoalEnv

from jueru.Agent_set import DDPG_agent
from jueru.algorithms import BaseAlgorithm, DQNAlgorithm, SACAlgorithm
from jueru.datacollection import Dict_Replay_buffer, Replay_buffer
from jueru.envs.uav_env.uav_env import Uav_env
from jueru.updator import actor_updator_ddpg, critic_updator_ddpg, soft_update
from jueru.user.custom_actor_critic import CombinedExtractor, ddpg_actor, ddpg_critic, FlattenExtractor
from jueru.utils import scan_root

from trick_ddpg import Trick_DDPG
from uav_env_contrast import Environment_2D

env = Environment_2D()

feature_dim = 128

feature_extractor = FlattenExtractor(env.observation_space)

actor = ddpg_actor(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

critic = ddpg_critic(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

data_collection_dict = {}

data_collection_dict['success_replay_buffer'] = Replay_buffer(env=env, size=2e6)

data_collection_dict['fail_replay_buffer'] = Replay_buffer(env=env, size=1e6)

functor_dict = {}

lr_dict = {}

updator_dict = {}

functor_dict['actor'] = actor

functor_dict['critic'] = critic

functor_dict['actor_target'] = None

functor_dict['critic_target'] = None

lr_dict['actor'] = 1e-4

lr_dict['critic'] = 1e-4

lr_dict['actor_target'] = 1e-3

lr_dict['critic_target'] = 1e-3

updator_dict['actor_update'] = actor_updator_ddpg

updator_dict['critic_update'] = critic_updator_ddpg

updator_dict['soft_update'] = soft_update

ddpg = Trick_DDPG(agent_class=DDPG_agent,
                  functor_dict=functor_dict,
                  lr_dict=lr_dict,
                  updator_dict=updator_dict,
                  data_collection_dict=data_collection_dict,
                  env=env,
                  gamma=0.95,
                  batch_size=100,
                  tensorboard_log="./Trick_DDPG_tensorboard/",
                  render=False,
                  action_noise=0.1,
                  min_update_step=1000,
                  update_step=100,
                  polyak=0.995,
                  start_steps=3000,
                  save_interval=2000,
                  model_address='./Trick_DDPG_model_address',
                  save_mode='eval',
                  eval_freq=30
                  )

ddpg.learn(num_train_step=330000, actor_update_freq=4, proportion=0.5)
