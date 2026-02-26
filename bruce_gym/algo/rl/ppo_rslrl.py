# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# Copyright (c) 2026 ECO Authors. All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_critic_rslrl import ActorCritic_RSLRL
from .rollout_storage import RolloutStorage

class PPO_RSLRL:
    actor_critic: ActorCritic_RSLRL
    def __init__(self,
                actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 cost_gamma=0.998,
                 cost_lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 use_cost_values=[],
                 **args,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.cost_value_optimizer = optim.Adam(self.actor_critic.cost_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_cost_values = use_cost_values

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.use_cost_values, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        rew_value = self.actor_critic.evaluate(critic_obs)
        cost_value = self.actor_critic.cost_evaluate(critic_obs)
        cost_value_num = 0
        for i in range(len(self.use_cost_values)):
            if self.use_cost_values[i]:
                setattr(self.transition, 'cost_values'+str(i+1), cost_value[:, cost_value_num].unsqueeze(1).detach())
                cost_value_num += 1
            else:
                setattr(self.transition, 'cost_values'+str(i+1), 0.0)
        self.transition.values = rew_value.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        for i in range(len(self.use_cost_values)):
            setattr(self.transition, 'costs'+str(i+1), costs[i].clone())

        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            for i in range(len(self.use_cost_values)):
                cost = getattr(self.transition, 'costs'+str(i+1))
                if self.use_cost_values[i]:
                    setattr(self.transition, 'costs'+str(i+1), cost + self.cost_gamma * torch.squeeze(getattr(self.transition, 'cost_values'+str(i+1)) * infos['time_outs'].unsqueeze(1).to(self.device), 1))

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs)
        last_cost_values = self.actor_critic.cost_evaluate(last_critic_obs)

        self.storage.compute_returns(last_values.detach(), self.gamma, self.lam)
        use_value_num = 0
        for i in range(len(self.use_cost_values)):
            if self.use_cost_values[i]:
                self.storage.compute_cost_returns(last_cost_values[:, use_value_num].unsqueeze(1), self.cost_gamma, self.cost_lam, i+1)
                use_value_num += 1
            else:
                self.storage.compute_cost_returns_novalue(self.cost_gamma, self.cost_lam, i+1)

    def update(self, lagrange, mirror_observation=None, mirror_action=None):
        mean_value_loss = 0
        mean_mirror_loss = 0
        mean_cost_value_loss = 0

        mean_surrogate_loss = 0

        mean_advantages = 0

        mean_cost_advantages = 0
        mean_ratio = 0
        max_ratio = 0
        min_ratio = 0
        mean_actions_log_prob = 0
        max_actions_log_prob = 0
        min_actions_log_prob = 0
        mean_old_actions_log_prob = 0
        max_old_actions_log_prob = 0
        min_old_actions_log_prob = 0
        mean_entropy = 0
        max_entropy = 0
        min_entropy = 0
        
        mean_reward_return = 0
        max_reward_return = 0
        min_reward_return = 0
        
        mean_reward_value = 0
        max_reward_value = 0
        min_reward_value = 0

        mean_kl = 0
        max_kl = 0
        min_kl = 0
        
        mean_sigma = 0
        max_sigma = 0
        min_sigma = 0
        
        mean_mu = 0
        max_mu = 0
        min_mu = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, cost_target_values_batch, advantages_batch, cost_advantages_batch,returns_batch, cost_returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                if mirror_observation is not None and mirror_action is not None:
                    deterministic_actions = self.actor_critic.act_inference(obs_batch)
                    mir_obs = mirror_observation(obs_batch)
                    mirror_actions = self.actor_critic.act_inference(mir_obs)
                    mirror_actions = mirror_action(mirror_actions)
                    mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
            
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])    
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                
                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    
                    
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
             
                # Gradient step

                if mirror_observation is not None:
                    loss = surrogate_loss
                    loss += mirror_loss*lagrange[-1]
                    loss += self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                else:
                    loss = surrogate_loss  + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()         


                
                ratio_mean = ratio.mean()
                ratio_max = ratio.max()
                ratio_min = ratio.min()
                actions_log_prob_mean = actions_log_prob_batch.mean()
                actions_log_prob_max = actions_log_prob_batch.max()
                actions_log_prob_min = actions_log_prob_batch.min()
                old_actions_log_prob_mean = old_actions_log_prob_batch.mean()
                old_actions_log_prob_max = old_actions_log_prob_batch.max()
                old_actions_log_prob_min = old_actions_log_prob_batch.min()
                entropy_mean = entropy_batch.mean()
                entropy_max = entropy_batch.max()
                entropy_min = entropy_batch.min()
                
                mean_kl += kl_mean.item()
                max_kl += kl_mean.max().item()
                min_kl += kl_mean.min().item()
                mean_sigma += sigma_batch.mean().item()
                max_sigma += sigma_batch.max().item()
                min_sigma += sigma_batch.min().item()
                mean_mu += mu_batch.mean().item()
                max_mu += mu_batch.max().item()
                min_mu += mu_batch.min().item()
                
                mean_reward_value += value_batch.mean().item()
                max_reward_value += value_batch.max().item()
                min_reward_value += value_batch.min().item()
                mean_reward_return += returns_batch.mean().item()
                max_reward_return += returns_batch.max().item()
                min_reward_return += returns_batch.min().item()
                
                mean_ratio += ratio_mean.item()
                max_ratio += ratio_max.item()
                min_ratio += ratio_min.item()
                mean_actions_log_prob += actions_log_prob_mean.item()
                max_actions_log_prob += actions_log_prob_max.item()
                min_actions_log_prob += actions_log_prob_min.item()
                mean_old_actions_log_prob += old_actions_log_prob_mean.item()
                max_old_actions_log_prob += old_actions_log_prob_max.item()
                min_old_actions_log_prob += old_actions_log_prob_min.item()
                mean_entropy += entropy_mean.item()
                max_entropy += entropy_max.item()
                min_entropy += entropy_min.item()

                mean_value_loss += value_loss.item()

                mean_mirror_loss += mirror_loss.item() if mirror_observation is not None else 0.0
                mean_surrogate_loss += surrogate_loss.item()
                
                mean_advantages += advantages_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_kl /= num_updates
        max_kl /= num_updates
        min_kl /= num_updates
        mean_sigma /= num_updates
        max_sigma /= num_updates
        min_sigma /= num_updates
        mean_mu /= num_updates
        max_mu /= num_updates
        min_mu /= num_updates
        
        mean_reward_return /= num_updates
        max_reward_return /= num_updates
        min_reward_return /= num_updates
        mean_reward_value /= num_updates
        max_reward_value /= num_updates
        min_reward_value /= num_updates
        mean_reward_value /= num_updates
        max_reward_value /= num_updates
        min_reward_value /= num_updates
        mean_ratio /= num_updates
        max_ratio /= num_updates
        min_ratio /= num_updates
        mean_actions_log_prob /= num_updates
        max_actions_log_prob /= num_updates
        min_actions_log_prob /= num_updates
        mean_old_actions_log_prob /= num_updates
        max_old_actions_log_prob /= num_updates
        min_old_actions_log_prob /= num_updates
        mean_entropy /= num_updates
        max_entropy /= num_updates
        min_entropy /= num_updates
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_mirror_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_advantages /= num_updates
        mean_cost_advantages /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_cost_value_loss, mean_surrogate_loss, mean_mirror_loss, mean_advantages, mean_cost_advantages, mean_reward_value, max_reward_value, min_reward_value, mean_reward_return, max_reward_return, min_reward_return, mean_ratio, max_ratio, min_ratio, mean_actions_log_prob, max_actions_log_prob, min_actions_log_prob, mean_old_actions_log_prob, max_old_actions_log_prob, min_old_actions_log_prob, mean_entropy, max_entropy, min_entropy, mean_kl, max_kl, min_kl, mean_sigma, max_sigma, min_sigma, mean_mu, max_mu, min_mu, 0.0, 0.0
