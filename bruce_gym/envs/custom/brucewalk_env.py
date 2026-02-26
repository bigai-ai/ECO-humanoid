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


from bruce_gym.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from bruce_gym.envs import LeggedRobot

from bruce_gym.utils.terrain import  HumanoidTerrain
# from collections import deque
from bruce_gym.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

# @ torch.jit.script
def random_sample(env_ids, low, high, device):
        """
        Generate random samples for each entry of env_ids
        """
        rand_pos = torch_rand_float(0, 1, (len(env_ids), len(low)),
                                    device=device)
        diff_pos = (high - low).repeat(len(env_ids),1)
        random_dof_pos = rand_pos*diff_pos + low.repeat(len(env_ids), 1)
        return random_dof_pos 


def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat

class BruceWalkFreeEnv(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.03
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()
        self.append_obs = [0.1, 1, 2, -3, -4] #clock, 3D command (vel_x, vel_y, aug_vel_yaw)
        base_mir_obs = [
                        -10, 11, -12, 13, 14, # pos
                        -5, 6, -7, 8, 9, # pos 
                        -20, 21, -22, 23, 24,  # vel
                        -15, 16, -17, 18, 19,  # vel
                        -30, 31, -32, 33, 34,  # last action 
                        -25, 26, -27, 28, 29,  # last action
                        -35, 36, -37, # ang 
                        -38, 39 # euler
                        ]
        self.clock_inds = self.append_obs[0:2]
        self.mirrored_obs = np.array(self.append_obs+base_mir_obs, copy=True).tolist()
        self.mirrored_act = [-5, 6, -7, 8, 9, 
                            -0.1, 1, -2, 3, 4]
        self.obs_mirror_matrix = torch.tensor(_get_symmetry_matrix(self.mirrored_obs), device=self.device, dtype=torch.float32)
        self.act_mirror_matrix = torch.tensor(_get_symmetry_matrix(self.mirrored_act), device=self.device, dtype=torch.float32)

    def mirror_action(self, action):
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        return obs @ self.obs_mirror_matrix

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs):
        # print("obs.shape = ", obs.shape)
        # print("obs_mirror_matrix.shape = ", self.obs_mirror_matrix.shape)
        mirror_obs_batch = torch.zeros_like(obs, device=self.device)
        history_len = self.cfg.env.frame_stack # FIX HISTORY-OF-STATES LENGTH TO 1 FOR NOW
        for block in range(history_len):
            obs_ = obs[:, self.cfg.env.num_single_obs*block : self.cfg.env.num_single_obs*(block+1)]
            mirror_obs = obs_ @ self.obs_mirror_matrix
            clock = mirror_obs[:, self.clock_inds]

            for i in range(np.shape(clock)[1]):
                mirror_obs[:, int(self.clock_inds[i])] = torch.sin(torch.arcsin(clock[:, i]) + torch.pi)
            mirror_obs_batch[:, self.cfg.env.num_single_obs*block : self.cfg.env.num_single_obs*(block+1)] = mirror_obs
        return mirror_obs_batch


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)
    
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        ankle_pos = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.hip_pos_scale
        scale_2 = self.cfg.rewards.knee_pos_scale 
        scale_3 = self.cfg.rewards.ankle_pos_scale

        # left stand
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0


        self.ref_dof_pos[:, 1] =  -(sin_pos_l) * scale_1
        self.ref_dof_pos[:, 3] =  (sin_pos_l) * scale_2
        self.ref_dof_pos[:, 4] =  -sin_pos_l * scale_3
        
        # right swing
        self.ref_dof_pos[:, 6] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 8] =  - sin_pos_r * scale_2
        self.ref_dof_pos[:, 9] =  sin_pos_r * scale_3


        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.0
        self.ref_action = self.ref_dof_pos
        self.ref_dof_pos += self.default_dof_pos

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 15] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 150.0
        noise_vec[16] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[17] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 50.0
        noise_vec[18] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[19] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 50.0
        noise_vec[20] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 150.0
        noise_vec[21] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[22] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 50.0
        noise_vec[23] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[24] = noise_scales.dof_vel * self.obs_scales.dof_vel# * 50.0
        noise_vec[25: 35] = 0.  # previous actions
        noise_vec[35: 38] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[38: 40] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec


    def step(self, actions):
        if self.cfg.env.use_only_ref_actions:
            actions = self.ref_action
        else:
            if self.cfg.env.use_ref_actions:
                actions = self.ref_action

        return super().step(actions)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.Kp_factors
        d_gains = self.d_gains * self.Kd_factors

        if self.cfg.env.use_only_ref_actions:
            torques = p_gains * (actions - self.dof_pos) - d_gains * self.dof_vel
        else:
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device)
            torques *= motor_strength_factors

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.cost1_buf = torch.sum(torch.abs(self.torques * self.dof_vel),dim=1)

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            q, # 10
            dq,  # 10
            self.actions,  # 10
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            diff,  # 10
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.friction_coeffs,  # 1
            self.body_mass / 2.5,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)

        base_euler_noise = torch.zeros((self.num_envs, 2), device=self.device)
        base_euler_noise[:] = self.base_euler_xyz[:,:2]
        base_euler_noise[:] += self.euler_rand[:,:2]
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 10D
            dq,  # 10D
            self.actions,   # 10D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            (base_euler_noise) * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0


# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.152, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.152)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.152, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.152)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = torch.concat((joint_diff[:, 0].unsqueeze(1), joint_diff[:, 2].unsqueeze(1)),dim=1)
        right_yaw_roll = torch.concat((joint_diff[:, 5].unsqueeze(1), joint_diff[:, 7].unsqueeze(1)),dim=1)
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)


    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.03)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.03
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques)*100., dim=1)

    def _reward_energy(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.abs(self.torques * self.dof_vel),dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
