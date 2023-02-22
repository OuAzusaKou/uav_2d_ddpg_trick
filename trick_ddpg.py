from typing import Dict, Any

import numpy as np
import torch
from jueru.algorithms import BaseAlgorithm
from jueru.utils import merge_batch


class Trick_DDPG(BaseAlgorithm):

    def __init__(self, agent_class, data_collection_dict: None, env: None, updator_dict=None, functor_dict=None,
                 optimizer_dict=None, lr_dict=None, exploration_rate: float = 0.1, exploration_start: float = 1,
                 exploration_end: float = 0.05, exploration_fraction: float = 0.2, polyak: float = 0.9,
                 agent_args: Dict[str, Any] = None, max_episode_steps=None, gamma: float = 0.95, batch_size: int = 512,
                 tensorboard_log: str = "./DQN_tensorboard/", tensorboard_log_name: str = "run", render: bool = False,
                 action_noise: float = 0.1, min_update_step: int = 1000, update_step: int = 100,
                 start_steps: int = 10000, model_address: str = "./Base_model_address", save_mode: str = 'step',
                 save_interval: int = 5000, eval_freq: int = 100, eval_num_episode: int = 10,
                 scan_amount=5):
        super().__init__(agent_class, data_collection_dict, env, updator_dict, functor_dict, optimizer_dict, lr_dict,
                         exploration_rate, exploration_start, exploration_end, exploration_fraction, polyak, agent_args,
                         max_episode_steps, gamma, batch_size, tensorboard_log, tensorboard_log_name, render,
                         action_noise, min_update_step, update_step, start_steps, model_address, save_mode,
                         save_interval, eval_freq, eval_num_episode)

        self.merge_batch = merge_batch
        self.scan_amount = scan_amount

    def learn(self, num_train_step, actor_update_freq, reward_scale=1, proportion=0.3):
        """

        """

        self.agent.functor_dict['actor'].train()
        self.agent.functor_dict['critic'].train()
        # self.agent.actor_target.train()
        # self.agent.critic_target.train()
        step = 0
        episode_num = 0
        average_reward_buf = - 1e6
        while step <= (num_train_step):

            state = self.env.reset()
            episode_reward = 0
            episode_step = 0
            replay_list = []

            while True:

                if self.render:
                    self.env.render()

                if step >= self.start_steps:
                    # action = self.agent.choose_action(state, self.action_noise)
                    noise_state = self.state_attack(state, self.action_noise)
                    action = self.agent.choose_action(noise_state, self.action_noise)
                    print(action)
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)

                done_value = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                replay_list.append(
                    dict(state=state, action=action, reward=reward, next_state=next_state, done=done_value))

                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                # print(state)
                # self.data_collection_dict['replay_buffer'].store(state, action, reward, next_state, done_value)

                state = next_state.copy()

                episode_reward += reward

                # if step >= self.min_update_step and step % self.update_step == 0:
                #

                step += 1

                if step >= self.min_update_step and step % self.save_interval == 0:
                    self.agent.save(address=self.model_address)
                if done:
                    if reward > 0:
                        for experience in replay_list:
                            self.data_collection_dict['success_replay_buffer'].store(*experience.values())
                    else:
                        for experience in replay_list:
                            self.data_collection_dict['fail_replay_buffer'].store(*experience.values())
                    for i in range(self.update_step):

                        batch1 = self.data_collection_dict['success_replay_buffer'].sample_batch(
                            int(self.batch_size * proportion))  # random sample batch
                        # print(batch1)
                        batch2 = self.data_collection_dict['fail_replay_buffer'].sample_batch(
                            int(self.batch_size * (1 - proportion)))  # random sample batch
                        batch = self.merge_batch(batch1, batch2)

                        critic_loss = self.updator_dict['critic_update'](self.agent, state=batch['state'],
                                                                         action=batch['action'],
                                                                         reward=batch['reward'],
                                                                         next_state=batch['next_state'],
                                                                         done_value=batch['done'], gamma=self.gamma)
                        if i % 4 == 0:
                            actor_loss = self.updator_dict['actor_update'](self.agent, state=batch['state'],
                                                                           action=batch['action'],
                                                                           reward=batch['reward'],
                                                                           next_state=batch['next_state'],
                                                                           gamma=self.gamma)

                            self.updator_dict['soft_update'](self.agent.functor_dict['actor_target'],
                                                             self.agent.functor_dict['actor'],
                                                             polyak=self.polyak)

                            self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'],
                                                             self.agent.functor_dict['critic'],
                                                             polyak=self.polyak)

                        self.writer.add_scalar('critic_loss', critic_loss, global_step=(step + i))
                        self.writer.add_scalar('actor_loss', actor_loss, global_step=(step + i))

                    episode_num += 1

                    self.writer.add_scalar('episode_reward', episode_reward, global_step=step)

                    if self.save_mode == 'eval':
                        if step >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance(num_episode=self.eval_num_episode, step=step)
                            if average_reward > average_reward_buf:
                                self.agent.save(address=self.model_address)
                            average_reward_buf = average_reward
                    break

    def state_attack(self, state, noise_scale):
        """

        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.agent.functor_dict['actor'](state_tensor)

        q_lower = self.agent.functor_dict['critic_target'](state_tensor, action)

        for i in range(self.scan_amount):

            noise_state = state + noise_scale * np.random.randn(*self.env.observation_space.shape)

            noise_state_tensor = torch.as_tensor(noise_state, dtype=torch.float32).unsqueeze(0)

            noise_action = self.agent.functor_dict['actor'](noise_state_tensor)

            q_noise = self.agent.functor_dict['critic_target'](noise_state_tensor, noise_action)

            if q_noise < q_lower:
                q_lower = q_noise

                state = noise_state

        return noise_state
