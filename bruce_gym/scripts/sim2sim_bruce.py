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

import csv
import time
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from bruce_gym import LEGGED_GYM_ROOT_DIR
from bruce_gym.envs import BruceWalkCfg
from bruce_gym.utils import  Logger
from bruce_gym.algo.rl.actor_critic_rslrl import ActorCritic_RSLRL
import torch

class cmd:
    vx = 0.1
    vy = 0.0
    dyaw = 0.0

def log_step(load_run,step, ckpt):
    log_file_path = "./log_file_mujoco_"+load_run+".log"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{ckpt}-{step}\n")
        
        
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    l_ankele_pos = data.body('ankle_pitch_l').xpos.copy()
    base_pos = data.body('bruce-pelvis').xpos.copy()

    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec, l_ankele_pos, base_pos)

def launch_ball(data, launch_pos, dir_vec):
    data.qpos[-7:] = np.array([
        launch_pos[0], launch_pos[1], launch_pos[2],
        1.0, 0.0, 0.0, 0.0
    ])
    data.qvel[-6:] = np.array([
        dir_vec[0]*3.0, dir_vec[1]*3.0, dir_vec[2]*3.0,
        0.0, 0.0, 0.0
    ])

# --- random launcher config ---
RADIUS_RANGE = (0.2, 0.8)   # m, distance from robot to spawn ball
HEIGHT_RANGE = (-0.2, 1.0)   # m, spawn height above ground
SPEED_RANGE  = (5.0, 8.0)   # m/s, initial speed toward robot
TARGET_Z_BIAS = 0.0         # m, aim roughly at torso (pelvis.z + bias)
RNG = np.random.default_rng()  # reproducible if you pass a seed

def set_freejoint_state_by_name(model, data, joint_name, pos_xyz, quat_wxyz, lin_vel_xyz, ang_vel_xyz=(0.0,0.0,0.0)):
    """Safe per-element assignment for a freejoint (works on DeepMind mujoco binding)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        # Fallback: assume ball is last in state (only if you purposely placed it last)
        data.qpos[-7:] = np.array([*pos_xyz, *quat_wxyz], dtype=float)
        data.qvel[-6:] = np.array([*lin_vel_xyz, *ang_vel_xyz], dtype=float)
        mujoco.mj_forward(model, data)
        return
    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])

    # qpos: [x y z qw qx qy qz]
    data.qpos[qadr + 0] = pos_xyz[0]
    data.qpos[qadr + 1] = pos_xyz[1]
    data.qpos[qadr + 2] = pos_xyz[2]
    data.qpos[qadr + 3] = quat_wxyz[0]
    data.qpos[qadr + 4] = quat_wxyz[1]
    data.qpos[qadr + 5] = quat_wxyz[2]
    data.qpos[qadr + 6] = quat_wxyz[3]

    # qvel: [vx vy vz wx wy wz]
    data.qvel[dadr + 0] = lin_vel_xyz[0]
    data.qvel[dadr + 1] = lin_vel_xyz[1]
    data.qvel[dadr + 2] = lin_vel_xyz[2]
    data.qvel[dadr + 3] = ang_vel_xyz[0]
    data.qvel[dadr + 4] = ang_vel_xyz[1]
    data.qvel[dadr + 5] = ang_vel_xyz[2]

    mujoco.mj_forward(model, data)

def sample_ball_shot(model, data, base_body="bruce-pelvis"):
    """Sample a random spawn position around the robot and a velocity aimed at the robot with random speed."""
    base_pos = data.body(base_body).xpos.copy()

    # random ring around robot on XY, random height
    r = RNG.uniform(*RADIUS_RANGE)
    theta = RNG.uniform(0.0, 2.0*np.pi)
    h = RNG.uniform(*HEIGHT_RANGE)

    spawn = base_pos + np.array([r*np.cos(theta), r*np.sin(theta), h])

    # aim slightly above pelvis center, to reduce ground hits
    aim = base_pos + np.array([0.0, 0.0, TARGET_Z_BIAS])

    dir_vec = aim - spawn
    nrm = np.linalg.norm(dir_vec)
    if nrm < 1e-8:  # degenerate (extremely unlikely)
        dir_vec = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    dir_unit = dir_vec / nrm

    speed = RNG.uniform(*SPEED_RANGE)
    vel = dir_unit * speed

    # keep the ball above ground and outside of robot body initially
    spawn[2] = max(spawn[2], 0.1)

    return spawn, vel

  
def pd_control(target_q, default_dof_pos, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q + default_dof_pos - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    csv_file_path = time.strftime('simulation_mujoco_data_' + cfg.sim_config.load_model.split('/')[-1]+'_%Y%m%d_%H%M%S.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['step', 'l_ankele_pos_z', 'lin_vel_x'])  # CSV header

        logger = Logger(0.01)

        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        #model.body_name2id('ankle_pitch_l')

        data = mujoco.MjData(model)
        # Retrieve ball joint/body IDs
        ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        ball_bodyid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # Starting indices in qpos/qvel for the ball free joint
        ball_jntadr = model.jnt_qposadr[ball_joint_id]
        ball_veladr = model.jnt_dofadr[ball_joint_id]

        steps_per_second = int(0.6 / cfg.sim_config.dt)
        #body_pos = data.xpos[model.body_name2id('ankle_pitch_l')]
        mujoco.mj_resetDataKeyframe(model, data, 0)

        mujoco.mj_step(model, data)
        viewer = mujoco_viewer.MujocoViewer(model, data)
        # Configure camera view
        viewer.cam.azimuth = 90  # azimuth
        viewer.cam.elevation = 0  # elevation
        viewer.cam.distance = 2.0  # distance to target
        viewer.cam.lookat[0] = 0.3  # target x
        viewer.cam.lookat[1] = -0.6  # target y
        viewer.cam.lookat[2] = 0.5  # target z
        target_q = np.zeros((cfg.env.num_actions+6), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 0

        default_dof_pos = np.array([0.0, 0.39, 0.0, -0.64, 0.3, 
                                    0.0, 0.39, 0.0, -0.64, 0.3,
                                    -0.7, 1.3, 2.0, 
                                    0.7, -1.3, -2.0])

        sum_energy = 0
        sum_lin_vel_x = 0
        num_steps = 0
        for step in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
            # Launch one ball every second
            if step % steps_per_second == 0:
                spawn_pos, init_vel = sample_ball_shot(model, data, base_body="bruce-pelvis")
                set_freejoint_state_by_name(
                    model, data,
                    joint_name="ball_freejoint",           # name implied by <freejoint/> under <body name="ball">
                    pos_xyz=spawn_pos,
                    quat_wxyz=(1.0, 0.0, 0.0, 0.0),        # unit quaternion
                    lin_vel_xyz=init_vel,
                    ang_vel_xyz=(0.0, 0.0, 0.0)
                )
            q, dq, quat, v, omega, gvec, l_ankele_pos, base_pos = get_obs(data)
            #print(model.body('ankle_pitch_l').pos)
            q = np.array(data.actuator_length)
            dq = np.array(data.actuator_velocity)
            if count_lowlevel % cfg.sim_config.decimation == 0:
                csv_writer.writerow([int(step/10)] + [l_ankele_pos[2]] + [v[0]])
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
                obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:15] = (q[:cfg.env.num_actions] - default_dof_pos[:cfg.env.num_actions])* cfg.normalization.obs_scales.dof_pos
                obs[0, 15:25] = dq[:cfg.env.num_actions] * cfg.normalization.obs_scales.dof_vel
                obs[0, 25:35] = action
                obs[0, 35:38] = omega
                obs[0, 38:40] = eu_ang[0:2]

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0,:10].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

                target_q[:cfg.env.num_actions] = action * cfg.control.action_scale

                viewer.render()

                true_target_q = target_q + default_dof_pos

                logger.log_states(
                    {

                        'dof_pos_target0_gym': true_target_q[0],
                        'dof_pos0_gym': q[0] ,
                        'dof_pos_target1_gym': true_target_q[1],
                        'dof_pos1_gym': q[1],
                        'dof_pos_target2_gym': true_target_q[2],
                        'dof_pos2_gym': q[2],
                        'dof_pos_target3_gym': true_target_q[3],
                        'dof_pos3_gym': q[3],
                        'dof_pos_target4_gym': true_target_q[4],
                        'dof_pos4_gym': q[4],
                        'dof_pos_target5_gym': true_target_q[5],
                        'dof_pos5_gym': q[5],
                        'dof_pos_target6_gym': true_target_q[6],
                        'dof_pos6_gym': q[6],
                        'dof_pos_target7_gym': true_target_q[7],
                        'dof_pos7_gym': q[7],
                        'dof_pos_target8_gym': true_target_q[8],
                        'dof_pos8_gym': q[8],
                        'dof_pos_target9_gym': true_target_q[9],
                        'dof_pos9_gym': q[9],
                        'base_ang0_gym': omega[0],
                        'base_ang1_gym': omega[1],
                        'base_ang2_gym': omega[2],

                        
                        'dof_vel0_gym': dq[0],
                        'dof_vel1_gym': dq[1],
                        'dof_vel2_gym': dq[2],
                        'dof_vel3_gym': dq[3],
                        'dof_vel4_gym': dq[4],
                        
                        'dof_vel5_gym': dq[5],
                        'dof_vel6_gym': dq[6],
                        'dof_vel7_gym': dq[7],
                        'dof_vel8_gym': dq[8],
                        'dof_vel9_gym': dq[9],

                        'command_x': cmd.vx,
                        'command_y': cmd.vy,
                        'command_yaw': cmd.dyaw,
                        'base_vel_x': v[0],
                        'base_vel_y': v[1],
                        'base_vel_z': v[2],
                        # 'pred_vel_x': vel[0].item(),
                        # 'pred_vel_y': vel[1].item(),
                        # 'pred_vel_z': vel[2].item(),

                        
                        'base_euler0_gym': eu_ang[0],
                        'base_euler1_gym': eu_ang[1],
                        'base_euler2_gym': eu_ang[2],
                        
                        
                        # 'base_ang0_gazebo': env_gazebo.base_ang_vel[robot_index, 0].item(),
                        # 'base_ang1_gazebo': env_gazebo.base_ang_vel[robot_index, 1].item(),
                        # 'base_ang2_gazebo': env_gazebo.base_ang_vel[robot_index, 2].item(),

                    }
                    )

            count_lowlevel += 1
            target_dq = np.zeros((cfg.env.num_actions+6), dtype=np.double)

            # Generate PD control
            tau = pd_control(target_q, default_dof_pos, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

            data.ctrl = tau
            if count_lowlevel % cfg.sim_config.decimation == 0:
                sum_energy += (np.sum(np.abs(tau[:cfg.env.num_actions]*dq[:cfg.env.num_actions])))
                sum_lin_vel_x += v[0]

            if step%5000 == 0:
                print("avg_lin_vel_x", sum_lin_vel_x/500, "avg_energy", sum_energy/500)
                sum_lin_vel_x = 0
                sum_energy = 0
            mujoco.mj_step(model, data)
            if step == 80000 or base_pos[2] < 0.2:
                print("sum_energy", sum_energy)
                break

        print("step", step)

        viewer.close()
        logger.plot_states()
        return step

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--checkpoint', type=int, required=False, default=0,
                        help='checkpoint to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(BruceWalkCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/bruce_mj_description/xml/bruce_description.xml'
            sim_duration = 300.0
            dt = 0.001
            decimation = 10
            load_model = args.load_model
        class robot_config:
            kps = np.array([7, 10, 7, 10, 1.5, 7, 10, 7, 10, 1.5, 5, 5, 5, 5, 5, 5], dtype=np.double)
            # kds = np.array([0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.2, 1, 1, 1, 1, 1, 1], dtype=np.double)
            kds = np.array([0.2, 0.4, 0.2, 0.4, 0.08, 0.2, 0.4, 0.2, 0.4, 0.08, 1, 1, 1, 1, 1, 1], dtype=np.double)

            tau_limit = 8.5 * np.ones(16, dtype=np.double)

    # policy = torch.jit.load(args.load_model)
    for i in range(args.checkpoint, 1000000, 100):
        policy_cfg = {
        'actor_hidden_dims':[512, 256, 128],
        'critic_hidden_dims':[768, 256, 128],
        }

        num_critic_obs = 65*3
        num_actions = 10
        frame_stack = 15
        num_single_obs = 40

        num_observations = frame_stack * num_single_obs

        actor_critic = ActorCritic_RSLRL(num_observations, num_critic_obs, num_actions,**policy_cfg).to('cpu')

        model_path = args.load_model + '/model_'+str(i)+'.pt'
        loaded_dict = torch.load(model_path, map_location='cuda:0')
        actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic.eval() # switch to evaluation mode (dropout for example)
        policy = actor_critic.act_inference
        step = run_mujoco(policy, Sim2simCfg())
        log_step(args.load_model.split('/')[-1], step, i)
        break
