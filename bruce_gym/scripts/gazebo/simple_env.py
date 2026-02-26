# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2026 ECO Authors. All rights reserved.

import math
from collections import deque

import numpy as np
import torch

import time
import Settings.BRUCE_data as RDS
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
from Settings.BRUCE_macros import *
from Library.BRUCE_GYM.GAZEBO_INTERFACE import Manager as gazint

#####
from isaacgym.torch_utils import *


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__modified_by__ = "ECO Authors"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__eco_copyright__ = "Copyright 2026 ECO Authors"
__date__      = "September 6, 2023"
__version__   = "0.0.3"
__status__    = "Production"

"""
Script for communication with Gazebo
"""

class GazeboSimulator:
    def __init__(self):
        # robot info
        self.num_legs = 2
        self.num_joints_per_leg = 5
        self.num_arms = 2
        self.num_joints_per_arms = 3
        self.num_joints = self.num_legs * self.num_joints_per_leg + self.num_arms * self.num_joints_per_arms
        self.num_contact_sensors = 4

        self.leg_p_gains = [7.0, 10.0, 7.0, 10.0, 1.5,
                            7.0, 10.0, 7.0, 10.0, 1.5
                            ]
        self.leg_i_gains = [  0,   0,   0,   0,     0,
                            0,   0,   0,   0,     0
                            ]
        self.leg_d_gains = [ 0.2, 0.4, 0.2, 0.4, 0.08,
                            0.2, 0.4, 0.2, 0.4, 0.08
                            ]
        

        # self.leg_p_gains = [20, 30,  20,  30,    10]
        # self.leg_i_gains = [  0,   0,   0,   0,     0]
        # self.leg_d_gains = [ 1., 1, 1.0,  1.0, 0.4]

        self.arm_p_gains = [ 1.6,  1.6,  1.6, 1.6,  1.6,  1.6]
        self.arm_i_gains = [   0,    0,    0,  0,    0,    0]
        self.arm_d_gains = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

        self.p_gains = self.leg_p_gains  + self.arm_p_gains  # the joint order matches the robot's sdf file
        self.i_gains = self.leg_i_gains + self.arm_i_gains 
        self.d_gains = self.leg_d_gains  + self.arm_d_gains 
        
        # simulator info
        self.simulator = None
        self.simulation_frequency = 1000  # Hz
        self.simulation_modes = {'torque': 0, 'position': 2}
        self.simulation_mode = self.simulation_modes['position']



    def initialize_simulator(self):

        self.simulator = gazint.GazeboInterface(robot_name='bruce', num_joints=self.num_joints, num_contact_sensors=self.num_contact_sensors)
        self.simulator.set_step_size(1. / self.simulation_frequency)
        self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_all_position_pid_gains(self.p_gains, self.i_gains, self.d_gains)

        # arm pose
        ar1, ar2, ar3 = -0.7,  1.3,  2.0
        al1, al2, al3 =  0.7, -1.3, -2.0

        # leg pose
        bpr = np.array([0.04, -0.07, -0.38])  # right foot position  in body frame
        bpl = np.array([0.04, +0.07, -0.38])  # left  foot position  in body frame
        bxr = np.array([1., 0., 0.])          # right foot direction in body frame
        bxl = np.array([1., 0., 0.])          # left  foot direction in body frame

        self.default_dof_pos = [0., 0.39, 0., -0.64, 0.3,
            0.,  0.39, 0., -0.64, 0.3,
                        ar1, ar2, ar3,
                        al1, al2, al3]

        print("self.initial_pose", self.default_dof_pos)

        self.simulator.reset_simulation(initial_pose=self.default_dof_pos)
        print('Gazebo Initialization Completed!')



    def write_position(self, leg_positions, arm_positions):
        """
        Send goal positions to the simulator.
        """
        goal_position = [leg_positions[0], leg_positions[1], leg_positions[2], leg_positions[3], leg_positions[4],
                         leg_positions[5], leg_positions[6], leg_positions[7], leg_positions[8], leg_positions[9],
                         arm_positions[0], arm_positions[1], arm_positions[2],
                         arm_positions[3], arm_positions[4], arm_positions[5]]
        if self.simulation_mode != self.simulation_modes['position']:
            self.simulation_mode = self.simulation_modes['position']
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_position(goal_position)

    def write_torque(self, leg_torques, arm_torques):
        """
        Send goal torques to the simulator.
        """
        goal_torque = [leg_torques[0], leg_torques[1], leg_torques[2], leg_torques[3], leg_torques[4],
                       leg_torques[5], leg_torques[6], leg_torques[7], leg_torques[8], leg_torques[9],
                       arm_torques[0], arm_torques[1], arm_torques[2],
                       arm_torques[3], arm_torques[4], arm_torques[5]]
        if self.simulation_mode != self.simulation_modes['torque']:
            self.simulation_mode = self.simulation_modes['torque']
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_torque(goal_torque)

    def get_arm_goal_torques(self, arm_positions, arm_velocities):
        """
        Calculate arm goal torques.
        """
        arm_goal_torque = np.zeros(6)
        for i in range(6):
            arm_goal_torque[i] = self.arm_p_gains[i % self.num_joints_per_arms] * (arm_positions[i] - self.q_arm[i]) + self.arm_d_gains[i % self.num_joints_per_arms] * (arm_velocities[i] - self.dq_arm[i])
        return arm_goal_torque
        
    def update_sensor_info(self):
        """
        Get sensor info and write it to shared memory.
        """
        # get joint states
        q = self.simulator.get_current_position()
        dq = self.simulator.get_current_velocity()
        force = self.simulator.get_current_force()
        self.q_leg  = np.array([q[0], q[1], q[2], q[3], q[4],
                                q[5], q[6], q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]
        self.dq_arm = dq[10:16]

        
        leg_data = {'joint_positions':  self.q_leg,
                    'joint_velocities': self.dq_leg,
                     'joint_torques': force[0:10],
                    }
        arm_data = {'joint_positions':  self.q_arm,
                    'joint_velocities': self.dq_arm}
        MM.LEG_STATE.set(leg_data)
        MM.ARM_STATE.set(arm_data)
        
        # get imu states
        self.accel = self.simulator.get_imu_acceleration()
        self.omega = self.simulator.get_imu_angular_rate()
        self.foot_contacts = self.simulator.get_foot_contacts()
        
        sense_data = {'imu_acceleration': self.accel,
                      'imu_ang_rate':     self.omega,
                      'foot_contacts':    self.foot_contacts}
        MM.SENSE_STATE.set(sense_data)

        
    def get_observation(self):
        """
        get observations write it to shared memory.
        """
        # get joint states
        q = self.simulator.get_current_position()
        dq = self.simulator.get_current_velocity()
        
        # TODO: fix joint order, leg joint states [10] and velocities [10]
        self.q_leg  = np.array([q[0], q[1], q[2], q[3], q[4],
                                q[5], q[6], q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]

        self.dq_arm = dq[10:16]

        # get body states
        self.accel = self.simulator.get_imu_acceleration()
        self.omega = self.simulator.get_imu_angular_rate()
        self.foot_contacts = self.simulator.get_foot_contacts()
        contact_states = np.array([
            1 if self.foot_contacts[0] or self.foot_contacts[1] else 0,
            1 if self.foot_contacts[2] or self.foot_contacts[3] else 0,
        ])

        R_wb = self.simulator.get_body_rot_mat()
        w_bb = R_wb.T @ self.omega
        p_wb = self.simulator.get_body_position()
        
        v_wb = self.simulator.get_body_velocity()
        a_wb = R_wb @ self.accel
        v_bb = R_wb.T @ v_wb
        proj_grav = R_wb.T @ np.array([0., 0., -1])

        quat = self.simulator.get_body_quaternion()
        obs = {
            "body_pos": p_wb,
            "body_linear_vel": v_bb,
            "body_angular_vel": w_bb,
            "proj_gravity": proj_grav,
            "leg_states": self.q_leg,
            "leg_velocities": self.dq_leg,
            "arm_states": self.q_arm,
            "arm_velocities": self.dq_arm,
            "contact_states": contact_states,   # [2] [RF, LF]
            "quat": quat,
        }

        return obs
    
    def calculate_robot_model(self):
        """
        Calculate kinematics & dynamics and write it to shared memory.
        """
        r1, r2, r3, r4, r5 = self.q_leg[0], self.q_leg[1], self.q_leg[2], self.q_leg[3], self.q_leg[4]
        l1, l2, l3, l4, l5 = self.q_leg[5], self.q_leg[6], self.q_leg[7], self.q_leg[8], self.q_leg[9]
        dr1, dr2, dr3, dr4, dr5 = self.dq_leg[0], self.dq_leg[1], self.dq_leg[2], self.dq_leg[3], self.dq_leg[4]
        dl1, dl2, dl3, dl4, dl5 = self.dq_leg[5], self.dq_leg[6], self.dq_leg[7], self.dq_leg[8], self.dq_leg[9]

        R_wb = self.simulator.get_body_rot_mat()
        w_bb = self.omega
        p_wb = self.simulator.get_body_position()
        v_wb = self.simulator.get_body_velocity()
        a_wb = R_wb @ self.accel
        v_bb = R_wb.T @ v_wb
        yaw_angle = np.arctan2(R_wb[1, 0], R_wb[0, 0])

        # compute leg forward kinematics
        p_bt_r, v_bt_r, Jv_bt_r, dJv_bt_r, \
        p_bh_r, v_bh_r, Jv_bh_r, dJv_bh_r, \
        p_ba_r, v_ba_r, Jv_ba_r, dJv_ba_r, \
        p_bf_r, v_bf_r,  R_bf_r,  Jw_bf_r, dJw_bf_r, \
        p_bt_l, v_bt_l, Jv_bt_l, dJv_bt_l, \
        p_bh_l, v_bh_l, Jv_bh_l, dJv_bh_l, \
        p_ba_l, v_ba_l, Jv_ba_l, dJv_ba_l, \
        p_bf_l, v_bf_l,  R_bf_l,  Jw_bf_l, dJw_bf_l = kin.legFK(r1, r2, r3, r4, r5,
                                                                l1, l2, l3, l4, l5,
                                                                dr1, dr2, dr3, dr4, dr5,
                                                                dl1, dl2, dl3, dl4, dl5)

        # compute robot forward kinematics
        p_wt_r, v_wt_r, Jv_wt_r, dJvdq_wt_r, \
        p_wh_r, v_wh_r, Jv_wh_r, dJvdq_wh_r, \
        p_wa_r, v_wa_r, Jv_wa_r, dJvdq_wa_r, \
        p_wf_r, v_wf_r,  \
        R_wf_r, w_ff_r, Jw_ff_r, dJwdq_ff_r, \
        p_wt_l, v_wt_l, Jv_wt_l, dJvdq_wt_l, \
        p_wh_l, v_wh_l, Jv_wh_l, dJvdq_wh_l, \
        p_wa_l, v_wa_l, Jv_wa_l, dJvdq_wa_l, \
        p_wf_l, v_wf_l,  \
        R_wf_l, w_ff_l, Jw_ff_l, dJwdq_ff_l = kin.robotFK(R_wb, p_wb, w_bb, v_bb,
                                                          p_bt_r, Jv_bt_r, dJv_bt_r,
                                                          p_bh_r, Jv_bh_r, dJv_bh_r,
                                                          p_ba_r, Jv_ba_r, dJv_ba_r, R_bf_r, Jw_bf_r, dJw_bf_r,
                                                          p_bt_l, Jv_bt_l, dJv_bt_l,
                                                          p_bh_l, Jv_bh_l, dJv_bh_l,
                                                          p_ba_l, Jv_ba_l, dJv_ba_l, R_bf_l, Jw_bf_l, dJw_bf_l,
                                                          dr1, dr2, dr3, dr4, dr5,
                                                          dl1, dl2, dl3, dl4, dl5)

        # calculate robot dynamics
        H, CG, AG, dAGdq, p_wg, v_wg, k_wg = dyn.robotID(R_wb, p_wb, w_bb, v_bb,
                                                         r1, r2, r3, r4, r5,
                                                         l1, l2, l3, l4, l5,
                                                         dr1, dr2, dr3, dr4, dr5,
                                                         dl1, dl2, dl3, dl4, dl5)

        # save as estimation data
        estimation_data = {}
        estimation_data['time_stamp']        = np.array([self.simulator.get_current_time()])
        estimation_data['body_position']     = p_wb
        estimation_data['body_velocity']     = v_wb
        estimation_data['body_acceleration'] = a_wb
        estimation_data['body_rot_matrix']   = R_wb
        estimation_data['body_ang_rate']     = w_bb
        estimation_data['body_yaw_ang']      = np.array([yaw_angle])
        estimation_data['com_position']      = p_wg
        estimation_data['com_velocity']      = v_wg
        estimation_data['ang_momentum']      = k_wg
        estimation_data['H_matrix']          = H
        estimation_data['CG_vector']         = CG
        estimation_data['AG_matrix']         = AG
        estimation_data['dAGdq_vector']      = dAGdq
        estimation_data['foot_contacts']     = self.foot_contacts

        estimation_data['right_foot_rot_matrix'] = R_wf_r
        estimation_data['right_foot_ang_rate']   = w_ff_r
        estimation_data['right_foot_Jw']         = Jw_ff_r
        estimation_data['right_foot_dJwdq']      = dJwdq_ff_r
        estimation_data['right_foot_position']   = p_wf_r
        estimation_data['right_foot_velocity']   = v_wf_r
        estimation_data['right_toe_position']    = p_wt_r
        estimation_data['right_toe_velocity']    = v_wt_r
        estimation_data['right_toe_Jv']          = Jv_wt_r
        estimation_data['right_toe_dJvdq']       = dJvdq_wt_r
        estimation_data['right_heel_position']   = p_wh_r
        estimation_data['right_heel_velocity']   = v_wh_r
        estimation_data['right_heel_Jv']         = Jv_wh_r
        estimation_data['right_heel_dJvdq']      = dJvdq_wh_r
        estimation_data['right_ankle_position']  = p_wa_r
        estimation_data['right_ankle_velocity']  = v_wa_r
        estimation_data['right_ankle_Jv']        = Jv_wa_r
        estimation_data['right_ankle_dJvdq']     = dJvdq_wa_r

        estimation_data['left_foot_rot_matrix']  = R_wf_l
        estimation_data['left_foot_ang_rate']    = w_ff_l
        estimation_data['left_foot_Jw']          = Jw_ff_l
        estimation_data['left_foot_dJwdq']       = dJwdq_ff_l
        estimation_data['left_foot_position']    = p_wf_l
        estimation_data['left_foot_velocity']    = v_wf_l
        estimation_data['left_toe_position']     = p_wt_l
        estimation_data['left_toe_velocity']     = v_wt_l
        estimation_data['left_toe_Jv']           = Jv_wt_l
        estimation_data['left_toe_dJvdq']        = dJvdq_wt_l
        estimation_data['left_heel_position']    = p_wh_l
        estimation_data['left_heel_velocity']    = v_wh_l
        estimation_data['left_heel_Jv']          = Jv_wh_l
        estimation_data['left_heel_dJvdq']       = dJvdq_wh_l
        estimation_data['left_ankle_position']   = p_wa_l
        estimation_data['left_ankle_velocity']   = v_wa_l
        estimation_data['left_ankle_Jv']         = Jv_wa_l
        estimation_data['left_ankle_dJvdq']      = dJvdq_wa_l

        MM.ESTIMATOR_STATE.set(estimation_data)

# Base class for RL tasks
class SimpleTask():

    def __init__(self, env_cfg):
        self.sim_params =   None
        self.physics_engine = None
        self.sim_device = 'cpu'
        self.headless = True
        self.dt = env_cfg.control.decimation * env_cfg.sim.dt
        self.cycle_time = 0.55
        self.decimation = env_cfg.control.decimation
        self.device =  'cpu'
        self.cfg = env_cfg
        self.graphics_device_id = -1
        self.num_envs = 1
        self.num_single_obs = 40
        self.num_observations_frame = 15
        self.num_obs = self.num_single_obs*self.num_observations_frame
        self.num_privileged_obs = (65)*3
        self.num_actions = 10
        
        self.torque_limits = 1000
        self.action_scale = env_cfg.control.action_scale
        self.actions = torch.zeros(1, self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(1, self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(1, self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, 16, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(16, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(16, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self.num_dof = 16
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_single_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}
        self.phase = torch.zeros(1, 1, dtype=torch.float,device=self.device, requires_grad=False)
        
        # create envs, sim and viewer


        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        self.command_torque = []

        self.gazebo_position = []
        self.gym_position = []
        self.desire_position = []

        self.gazebo_velocity = []
        self.gym_velocity = []
        self.body_vel_sum = 0.0
        self.energy_sum = 0.0
        # When restart this thread, reset the shared memory (so the robot is in idle)
        MM.init()
        MM.connect()

        # BRUCE SETUP
        self.Bruce = RDS.BRUCE()

        self.gs = GazeboSimulator()
        self.gs.initialize_simulator()


        MM.THREAD_STATE.set({'simulation': np.array([1.0])}, opt='only')  # thread is running
        
        self.phase = torch.zeros(1, 1, dtype=torch.float,device=self.device, requires_grad=False)
        self.hist_obs = deque()
        for _ in range(self.num_observations_frame):
            self.hist_obs.append(torch.zeros(1, int(self.num_single_obs), dtype=torch.float))
        self.obs_scales = self.cfg.normalization.obs_scales
        self.default_dof_pos = torch.tensor(self.gs.default_dof_pos, device=self.device).unsqueeze(0)

    def get_observations(self, action, velocity):
        
        obs = torch.zeros(
            1, self.num_single_obs, dtype=torch.float,
            device=self.device, requires_grad=False)

        gs_observation = self.gs.get_observation()

        leg_position_now = gs_observation['leg_states']
        HIP_YAW_R     = leg_position_now[0]
        HIP_PITCH_R = leg_position_now[1]
        HIP_ROLL_R= leg_position_now[2]

        KNEE_PITCH_R= leg_position_now[3]
        ANKLE_PITCH_R= leg_position_now[4]
        HIP_YAW_L   = leg_position_now[5]
        HIP_PITCH_L = leg_position_now[6]
        HIP_ROLL_L  = leg_position_now[7]

        KNEE_PITCH_L= leg_position_now[8]
        ANKLE_PITCH_L= leg_position_now[9]


        # arm_position_now = gs_observation['arm_states']
        # SHOULDER_PITCH_R = arm_position_now[0]
        # SHOULDER_ROLL_R = arm_position_now[1]
        # ELBOW_YAW_R = arm_position_now[2]
        # SHOULDER_PITCH_L = arm_position_now[3]
        # SHOULDER_ROLL_L = arm_position_now[4]
        # ELBOW_YAW_L = arm_position_now[5]

        joint_positions = torch.tensor(
            [[HIP_YAW_L, 
                HIP_PITCH_L,
                HIP_ROLL_L, 
                KNEE_PITCH_L,
                ANKLE_PITCH_L,
                HIP_YAW_R,
                HIP_PITCH_R,
                HIP_ROLL_R,
                KNEE_PITCH_R,
                ANKLE_PITCH_R,
                # SHOULDER_PITCH_L,
                # SHOULDER_ROLL_L,
                # ELBOW_YAW_L,
                # SHOULDER_PITCH_R,
                # SHOULDER_ROLL_R,
                # ELBOW_YAW_R,
                ]]
        ,dtype=torch.float,device=self.device, requires_grad=False)    


    
        leg_velocity_now = gs_observation['leg_velocities']

        HIP_YAW_R     = leg_velocity_now[0] 
        HIP_PITCH_R = leg_velocity_now[1]
        HIP_ROLL_R= leg_velocity_now[2] 
        KNEE_PITCH_R= leg_velocity_now[3]
        ANKLE_PITCH_R= leg_velocity_now[4]
        HIP_YAW_L   = leg_velocity_now[5]
        HIP_PITCH_L = leg_velocity_now[6]
        HIP_ROLL_L  = leg_velocity_now[7]
        KNEE_PITCH_L= leg_velocity_now[8]
        ANKLE_PITCH_L= leg_velocity_now[9]
        
        # arm_velocity_now = gs_observation['arm_states']
        # SHOULDER_PITCH_R = arm_velocity_now[0]
        # SHOULDER_ROLL_R = arm_velocity_now[1]
        # ELBOW_YAW_R = arm_velocity_now[2]
        # SHOULDER_PITCH_L = arm_velocity_now[3]
        # SHOULDER_ROLL_L = arm_velocity_now[4]
        # ELBOW_YAW_L = arm_velocity_now[5]
        
        joint_velocities =torch.tensor(
            [[HIP_YAW_L,
                HIP_PITCH_L, 
                HIP_ROLL_L, 
                KNEE_PITCH_L,
                ANKLE_PITCH_L,
                HIP_YAW_R,
                HIP_PITCH_R,
                HIP_ROLL_R,
                KNEE_PITCH_R,
                ANKLE_PITCH_R,
                # SHOULDER_PITCH_L,
                # SHOULDER_ROLL_L,
                # ELBOW_YAW_L,
                # SHOULDER_PITCH_R,
                # SHOULDER_ROLL_R,
                # ELBOW_YAW_R,
                ]]
        ,dtype=torch.float,device=self.device, requires_grad=False)
        
        self.dof_pos = joint_positions
        self.dof_vel = joint_velocities
    
        q = (joint_positions - self.default_dof_pos[:,:10]) * self.obs_scales.dof_pos
        
        
        obs[:,0] = torch.tensor(math.sin(2 * math.pi * self.phase * self.dt  / self.cycle_time),dtype=torch.float,device=self.device, requires_grad=False)
        
        obs[:,1] = torch.tensor(math.cos(2 * math.pi * self.phase * self.dt  / self.cycle_time) ,dtype=torch.float,device=self.device, requires_grad=False)
        obs[:,2] = torch.tensor([velocity[0]], dtype=torch.float,device=self.device, requires_grad=False) * self.cfg.normalization.obs_scales.lin_vel
        obs[:,3] = torch.tensor([velocity[1]], dtype=torch.float,device=self.device, requires_grad=False) * self.cfg.normalization.obs_scales.lin_vel 
        obs[:,4] = torch.tensor([velocity[2]], dtype=torch.float,device=self.device, requires_grad=False) * self.cfg.normalization.obs_scales.ang_vel 
        obs[:,5:15] = q
        obs[:,15:25] = joint_velocities * self.cfg.normalization.obs_scales.dof_vel
        obs[:, 25:35] = action
        obs[:, 35:38] = torch.tensor(gs_observation['body_angular_vel'],dtype=torch.float,device=self.device, requires_grad=False)  * self.obs_scales.ang_vel
        euler_xyz = get_euler_xyz_tensor(torch.tensor(gs_observation['quat'],dtype=torch.float,device=self.device, requires_grad=False).unsqueeze(0))
        obs[:, 38:40] = euler_xyz[:,:2][0] * self.obs_scales.quat

        self.hist_obs.append(obs)
        self.hist_obs.popleft()
        
        policy_input = torch.zeros(1, self.num_obs, dtype=torch.float, device=self.device)
        for i in range(self.num_observations_frame):
            policy_input[0, i * self.num_single_obs : (i + 1) * self.num_single_obs] = self.hist_obs[i][0, :]

        if gs_observation['body_pos'][2] < 0.3:
            slip = True
        else:
            slip = False

        return policy_input, slip
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""

        return None

    def reset(self):

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None
    def step(self, actions, velocity):

        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        position = (actions).cpu().numpy()[0]

        goal_positions = np.array([
                                position[5],
                                position[6],
                                position[7],
                                position[8],
                                position[9],
                                position[0],
                                position[1],
                                position[2],
                                position[3],
                                position[4],
                                # position[13],
                                # position[14],
                                # position[15],
                                # position[10],
                                # position[11],
                                # position[12],
                                    ])

        default_pos = np.array(
            self.gs.default_dof_pos
        )


        target_q = goal_positions * self.cfg.control.action_scale
        target_q = target_q + default_pos[:10]

        if self.Bruce.thread_error():
            self.Bruce.stop_threading()

        for _ in range(self.decimation):

            self.gs.update_sensor_info()
            self.gs.calculate_robot_model()

            gs_observation = self.gs.get_observation()

            leg_position_now = gs_observation['leg_states']
            leg_velocity_now = gs_observation['leg_velocities']

            leg_torques = self.gs.leg_p_gains*(target_q - leg_position_now) - self.gs.leg_d_gains*leg_velocity_now
            arm_position_now = gs_observation['arm_states']
            arm_velocity_now = gs_observation['arm_velocities']
            arm_torques = self.gs.arm_p_gains*(default_pos[10:] - arm_position_now) - self.gs.arm_d_gains*arm_velocity_now

            self.gs.write_torque(leg_torques, arm_torques)

            self.gs.simulator.step_simulation()
            time.sleep(0.000)

        self.gs.update_sensor_info()
        self.gs.calculate_robot_model()

        policy_input, slip = self.get_observations(actions.clone(), velocity)
        self.actions = actions.clone()

        self.phase += 1.0
        self.body_vel_sum += gs_observation['body_linear_vel'][0]
        self.energy_sum += sum(abs(leg_torques * gs_observation['leg_velocities']))

        if self.phase % 500 ==0: #0.01*500=5s
            print(self.phase, self.body_vel_sum / 500, self.energy_sum / 500)
            self.body_vel_sum = 0
            self.energy_sum = 0
        if self.phase % 1000==0:
            self.phase = 0
        return policy_input, slip







