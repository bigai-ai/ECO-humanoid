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

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.costs = None

            self.dones = None
            self.values = None
            self.cost_values = None

            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, use_cost_value, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.use_cost_value = use_cost_value
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        for i in range(len(use_cost_value)):
            setattr(self, 'costs' + str(i+1), torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device))
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        for i in range(len(use_cost_value)):
            setattr(self, 'cost_values' + str(i+1), torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device))
            setattr(self, 'cost_returns' + str(i+1), torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device))
            setattr(self, 'cost_advantages' + str(i+1), torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device))
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        for i in range(len(self.use_cost_value)):
            getattr(self, 'costs' + str(i+1))[self.step].copy_(getattr(transition, 'costs' + str(i+1)).view(-1, 1))
            getattr(self, 'cost_values' + str(i+1))[self.step].copy_(getattr(transition, 'cost_values' + str(i+1)))

        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_cost_returns_novalue(self, gamma, lam, cost_index):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = getattr(self, 'costs' + str(cost_index))[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            getattr(self, 'cost_returns' + str(cost_index))[step] = advantage
        # Compute and normalize the advantages
        setattr(self, 'cost_advantages' + str(cost_index), getattr(self, 'cost_returns'+str(cost_index)))
        setattr(self, 'cost_advantages' + str(cost_index),  (getattr(self, 'cost_advantages' + str(cost_index)) - getattr(self, 'cost_advantages' + str(cost_index)).mean()) / (getattr(self, 'cost_advantages' + str(cost_index)).std() + 1e-8))

    def compute_cost_returns(self, last_values, gamma, lam, cost_index):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = getattr(self, 'cost_values' + str(cost_index))[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = getattr(self, 'costs' + str(cost_index))[step] + next_is_not_terminal * gamma * next_values - getattr(self, 'cost_values' + str(cost_index))[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            getattr(self, 'cost_returns' + str(cost_index))[step] = advantage + getattr(self, 'cost_values' + str(cost_index))[step]
        # Compute and normalize the advantages
        setattr(self, 'cost_advantages' + str(cost_index), getattr(self, 'cost_returns'+str(cost_index)) - getattr(self, 'cost_values'+str(cost_index)))
        setattr(self, 'cost_advantages' + str(cost_index),  (getattr(self, 'cost_advantages' + str(cost_index)) - getattr(self, 'cost_advantages' + str(cost_index)).mean()) / (getattr(self, 'cost_advantages' + str(cost_index)).std() + 1e-8))

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean(), self.costs.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        
        ### TODO
        
        
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        for i in range(len(self.use_cost_value)):
            locals()['cost_values' + str(i+1)] = getattr(self, 'cost_values' + str(i+1)).flatten(0, 1)
            locals()['cost_returns' + str(i+1)] = getattr(self, 'cost_returns' + str(i+1)).flatten(0, 1)
            locals()['cost_advantages' + str(i+1)] = getattr(self, 'cost_advantages' + str(i+1)).flatten(0, 1)
        
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                target_cost_values_batch = []
                for i in range(len(self.use_cost_value)):
                    if self.use_cost_value[i]:
                        target_cost_values_batch.append(locals()['cost_values' + str(i+1)][batch_idx])
                if sum(self.use_cost_value)>0:
                    target_cost_values_batch = torch.concat(target_cost_values_batch, dim=1)
                else:
                    target_cost_values_batch = None
                returns_batch = returns[batch_idx]
                cost_returns_batch = []
                for i in range(len(self.use_cost_value)):
                    if self.use_cost_value[i]:
                        cost_returns_batch.append(locals()['cost_returns' + str(i+1)][batch_idx])
                if sum(self.use_cost_value)>0:
                    cost_returns_batch = torch.concat(cost_returns_batch, dim=1)
                else:
                    cost_returns_batch = None
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                cost_advantages_batch = []
                for i in range(len(self.use_cost_value)):
                    cost_advantages_batch.append(locals()['cost_advantages' + str(i+1)][batch_idx])
                if sum(self.use_cost_value)>0:
                    cost_advantages_batch = torch.concat(cost_advantages_batch, dim=1)
                else:
                    if cost_advantages_batch != []:
                        cost_advantages_batch = cost_advantages_batch[0].unsqueeze(1)
                    else:
                        cost_advantages_batch = None

                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, target_cost_values_batch, advantages_batch, cost_advantages_batch, returns_batch, cost_returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
