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

import os
import time
import torch
#import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo_rslrl import PPO_RSLRL
from .ppo_lag import PPOLag
from .p3o import P3O
from .crpo import CRPO

from .lag import  Lagrange
from .lag_ipo import IPOLagrange
from .lag_p3o import P3OLagrange
from .lag_crpo import CRPOLagrange

from .actor_critic_rslrl import ActorCritic_RSLRL
from bruce_gym.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter


from bruce_gym import LEGGED_GYM_ROOT_DIR

import os
import shutil

#source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
source_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'bruce_gym', 'envs', 'custom')

class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        self.all_cfg = train_cfg
        # self.wandb_run_name = (
        #     datetime.now().strftime("%b%d_%H-%M-%S")
        #     + "_"
        #     + train_cfg["runner"]["experiment_name"]
        #     + "_" 
        #     + train_cfg["runner"]["run_name"]
        # )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class: ActorCritic_RSLRL = eval(self.cfg["policy_class_name"])  # ActorCritic
        if hasattr(self.alg_cfg, "use_cost_values"):
            critic_output_dim = sum(self.alg_cfg['use_cost_values'])
        else:
            critic_output_dim = 1
        print("self.cfg['policy_class_name']", self.cfg["policy_class_name"])
        actor_critic: ActorCritic_RSLRL = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, critic_output_dim=critic_output_dim,**self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        lag_class = eval(self.cfg["lagrange_class_name"])

        self.alg: PPOLag|P3O|CRPO|PPO_RSLRL = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        for i in range(len(self.alg.use_cost_values)):
            setattr(self, "lagrange" + str(i+1),  lag_class(i+1,**self.alg_cfg))
        self.constraint_num = len(self.alg.use_cost_values)      
        setattr(self, "lagrange" + str(len(self.alg.use_cost_values)+1),  lag_class(len(self.alg.use_cost_values)+1,**self.alg_cfg)) 
        self.constraint_num += 1 # mirror loss
        ## At least, we use one constraint for the mirror loss

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]



        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.smooth_cost_mean = 0
        _, _ = self.env.reset()
        self.last_update = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # wandb.init(
            #     project="XBot",
            #     sync_tensorboard=True,
            #     name=self.wandb_run_name,
            #     config=self.all_cfg,
            # )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            source_file = source_dir + '/' + self.cfg["save_config"]
            shutil.copy(source_file, self.log_dir+'/' + self.cfg["save_config"])
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rewbuffer_idx = 0
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        for i in range(len(self.alg.use_cost_values)):
            locals()['cur_cost_sum'+str(i+1)] = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
            locals()['costbuffer'+str(i+1)] = deque(maxlen=100)
            
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            self.alg.actor_critic.eval()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, costs = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    cost_list = []
                    for i in range(len(costs)):
                        cost_list.append(costs[i].to(self.device))
                    self.alg.process_env_step(rewards, cost_list, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        for i in range(len(self.alg.use_cost_values)):
                            locals()['cur_cost_sum'+str(i+1)] += costs[i]        

                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        rewbuffer_idx += new_ids.shape[0]
                        for i in range(len(self.alg.use_cost_values)):
                            locals()['costbuffer'+str(i+1)].extend(locals()['cur_cost_sum'+str(i+1)][new_ids][:, 0].cpu().numpy().tolist())


                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        for i in range(len(self.alg.use_cost_values)):
                            locals()['cur_cost_sum'+str(i+1)][new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            self.alg.actor_critic.train()
            lagrange_items = []
            for i in range(self.constraint_num):
                lagrange_attr = 'lagrange'+str(i+1)
                lagrange_obj = getattr(self, lagrange_attr)
                if lagrange_obj is not None:
                    lagrange_items.append(lagrange_obj.lagrangian_multiplier.item())
            print("lagrange_items", lagrange_items)
            mean_value_loss, mean_cost_value_loss, mean_surrogate_loss, mean_mirror_loss, mean_advantages, mean_cost_advantages, mean_reward_value, max_reward_value, min_reward_value, mean_reward_return, max_reward_return, min_reward_return, mean_ratio, max_ratio, min_ratio, mean_actions_log_prob, max_actions_log_prob, min_actions_log_prob, mean_old_actions_log_prob, max_old_actions_log_prob, min_old_actions_log_prob, mean_entropy, max_entropy, min_entropy, mean_kl, max_kl, min_kl, mean_sigma, max_sigma, min_sigma, mean_mu, max_mu, min_mu, mean_alpha, mean_alpha_loss = self.alg.update(
                lagrange_items,
                self.env.mirror_clock_observation if hasattr(self.env, 'mirror_clock_observation') else None, 
                self.env.mirror_action if hasattr(self.env, 'mirror_action') else None
                )
            if len(self.alg.use_cost_values) > 1:
                for i in range(len(self.alg.use_cost_values)):
                    locals()['mean_cost_advantages'+str(i+1)] = mean_cost_advantages[int(i)]
            else:
                locals()['mean_cost_advantages'+str(1)] = mean_cost_advantages
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(it, os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            it, 
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if key == "rew_tracking_lin_vel":
                    rew_tracking_lin_vel = value.item()
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/alpha", locs["mean_alpha"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/alpha_loss", locs["mean_alpha_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Train/mean_mirror_loss", locs["mean_mirror_loss"], locs["it"]
        )
        if hasattr(self, 'lagrange' + str(len(self.alg.use_cost_values)+1)):
            getattr(self, 'lagrange' + str(len(self.alg.use_cost_values)+1)).update_lagrange_multiplier(locs["mean_mirror_loss"])
            self.writer.add_scalar(
                "Train/Lagrange_mirror",
            getattr(self, 'lagrange' + str(len(self.alg.use_cost_values)+1)).lagrangian_multiplier.item(),
                locs["it"],
        )
        self.writer.add_scalar(
            "Loss/cost_value_function", locs['mean_cost_value_loss'], locs["it"]
        )
        self.writer.add_scalar(
            "adv/mean_advantages", locs["mean_advantages"], locs["it"]
        )
        for i in range(len(self.alg.use_cost_values)):
            self.writer.add_scalar(
                "adv/mean_cost_advantages"+str(i+1), locs["mean_cost_advantages"+str(i+1)], locs["it"]
            )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        
        self.writer.add_scalar(
            "Debug/mean_kl", locs["mean_kl"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_kl", locs["max_kl"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_kl", locs["min_kl"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_sigma", locs["mean_sigma"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_sigma", locs["max_sigma"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_sigma", locs["min_sigma"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_mu", locs["mean_mu"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_mu", locs["max_mu"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_mu", locs["min_mu"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_ratio", locs["mean_ratio"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_ratio", locs["max_ratio"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_ratio", locs["min_ratio"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_actions_log_prob", locs["mean_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_actions_log_prob", locs["max_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_actions_log_prob", locs["min_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_old_actions_log_prob", locs["mean_old_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_old_actions_log_prob", locs["max_old_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_old_actions_log_prob", locs["min_old_actions_log_prob"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_entropy", locs["mean_entropy"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_entropy", locs["max_entropy"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_entropy", locs["min_entropy"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_reward_return", locs["mean_reward_return"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_reward_return", locs["max_reward_return"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_reward_return", locs["min_reward_return"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/mean_reward_value", locs["mean_reward_value"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/max_reward_value", locs["max_reward_value"], locs["it"]
        )
        self.writer.add_scalar(
            "Debug/min_reward_value", locs["min_reward_value"], locs["it"]
        )

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )
        for i in range(len(self.alg.use_cost_values)):
            if len(locs["costbuffer" + str(i+1)]) > 0 :
                self.writer.add_scalar(
                    "Train/mean_cost"+ str(i+1), statistics.mean(locs["costbuffer"+ str(i+1)]), locs["it"]
                )

                self.writer.add_scalar(
                    "Train/mean_cost"+ str(i+1) + "/time",
                    statistics.mean(locs["costbuffer"+ str(i+1)]),
                    self.tot_time,
                )

                self.writer.add_scalar(
                    "Train/Lagrange"+str(i+1),
                    getattr(self, "lagrange"+str(i+1)).lagrangian_multiplier.item(),
                    locs["it"],
                )
                getattr(self, "lagrange"+str(i+1)).update_lagrange_multiplier(statistics.mean(locs["costbuffer"+ str(i+1)]))

        str_log = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_log.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mirror loss:':>{pad}} {locs['mean_mirror_loss']:.4f}\n"""

                f"""{'Mean advantages:':>{pad}} {locs['mean_advantages']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_log.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mirror loss:':>{pad}} {locs['mean_mirror_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")


        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, it, path, infos=None):
        save_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "cost_value_optimizer_state_dict": self.alg.cost_value_optimizer.state_dict(),
            "iter": it,
            "total_time": self.tot_time,
            "infos": infos,
        }
        if hasattr(self, 'lagrange' + str(len(self.alg.use_cost_values)+1)):
            for i in range(len(self.alg.use_cost_values)+1):
                save_dict["lagrange"+str(i+1)] = getattr(self, "lagrange"+str(i+1)).lagrangian_multiplier
                if hasattr(getattr(self, "lagrange"+str(i+1)), "lambda_optimizer"):
                    save_dict["lagrange"+str(i+1) + "_lambda_optimizer"] = getattr(self, "lagrange"+str(i+1)).lambda_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location="cuda:0")
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.cost_value_optimizer.load_state_dict(loaded_dict["cost_value_optimizer_state_dict"])
            if "lagrange_lambda_optimizer" in loaded_dict and hasattr(self.lagrange, "lambda_optimizer"):
                self.lagrange.lambda_optimizer.load_state_dict(loaded_dict["lagrange_lambda_optimizer"])
            if "lagrange2_lambda_optimizer" in loaded_dict and hasattr(self.lagrange2, "lambda_optimizer"):
                self.lagrange2.lambda_optimizer.load_state_dict(loaded_dict["lagrange2_lambda_optimizer"])
        if "lagrange_lagrangian_multiplier" in loaded_dict:
            self.lagrange.lagrangian_multiplier = loaded_dict["lagrange_lagrangian_multiplier"]
        if "lagrange2_lagrangian_multiplier" in loaded_dict:
            self.lagrange2.lagrangian_multiplier = loaded_dict["lagrange2_lagrangian_multiplier"]
        if "total_time" in loaded_dict:
            self.tot_time = loaded_dict["total_time"]
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"] + 1
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
    
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
