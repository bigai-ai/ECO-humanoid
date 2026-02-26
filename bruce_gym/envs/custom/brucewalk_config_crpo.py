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


from bruce_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BruceWalkCRPOCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 40
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 65
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 8192
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False
        use_only_ref_actions =   False
        cost_limit1 = 6000
        cost_limit2 = 0.05
        env_cost_num = 1

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/bruce.urdf'

        name = "bruce"
        foot_name = "ankle_pitch"
        knee_name = "knee"

        terminate_after_contacts_on = [
            'base_link',
            'hip_yaw_link_l',
            'hip_roll_link_l',
            'hip_pitch_link_l',
            'knee_pitch_link_l',
            'hip_yaw_link_r',
            'hip_roll_link_r',
            'hip_pitch_link_r',
            'knee_pitch_link_r',
            'shoulder_pitch_link_l',
            'shoulder_roll_link_l',
            'elbow_pitch_link_l',
            'shoulder_pitch_link_r',
            'shoulder_roll_link_r',
            'elbow_pitch_link_r',
        ]

        #terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = [
            'base_link',
            'hip_yaw_link_l',
            'hip_roll_link_l',
            'hip_pitch_link_l',
            'knee_pitch_link_l',
            'hip_yaw_link_r',
            'hip_roll_link_r',
            'hip_pitch_link_r',
            'knee_pitch_link_r',
            'shoulder_pitch_link_l',
            'shoulder_roll_link_l',
            'elbow_pitch_link_l',
            'shoulder_pitch_link_r',
            'shoulder_roll_link_r',
            'elbow_pitch_link_r',
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' 
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.
     
    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.0
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.47]

        default_joint_angles = {
            'hip_yaw_l': 0.,
            'hip_pitch_l': 0.39,
            'hip_roll_l': 0.,
            'knee_pitch_l': -0.64,
            'ankle_pitch_l':0.3,
            'hip_yaw_r':0.,
            'hip_pitch_r': 0.39,
            'hip_roll_r':  0.,
            'knee_pitch_r':  -0.64,
            'ankle_pitch_r': 0.3,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_yaw_l': 7, 
            'hip_pitch_l': 10, 
            'hip_roll_l': 7, 
            'knee_pitch_l': 10, 
            'ankle_pitch_l': 1.5, 
            'hip_yaw_r': 7, 
            'hip_pitch_r': 10, 
            'hip_roll_r': 7, 
            'knee_pitch_r': 10, 
            'ankle_pitch_r': 1.5,
          }
        damping = {
            'hip_yaw_l': 0.2, 
            'hip_pitch_l': 0.4, 
            'hip_roll_l': 0.2, 
            'knee_pitch_l': 0.4, 
            'ankle_pitch_l': 0.08, 
            'hip_yaw_r': 0.2, 
            'hip_pitch_r': 0.4, 
            'hip_roll_r': 0.2, 
            'knee_pitch_r': 0.4, 
            'ankle_pitch_r': 0.08,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_dof_init = True
        dof_vel_init_range = [-0.15, 0.15]
        dof_pos_init_range = [-0.15, 0.15]
        
        randomize_payload_mass = True
        payload_mass_range = [-0.5, 0.5]

        randomize_euler = True
        euler_range = [-0.1, 0.1]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_restitution = True
        restitution_range = [0., 0.5]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_joint_friction = True
        joint_friction_range = [0.02, 0.05]

        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.01]

        disturbance = True
        disturbance_range = [-100.0, 100.0]
        disturbance_s = 2

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4

        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]

        delay = True
        delay_range = [1, 8]
        
    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.1, 0.1]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.45
        min_dist = 0.13
        max_dist = 0.2
         
        hip_pos_scale = 0.3    # rad
        knee_pos_scale = 0.6
        ankle_pos_scale = 0.3
        
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.03       # m
        cycle_time = 0.55                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 10
        max_contact_force = 50  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 0.0
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # smoothness
            action_smoothness = -0.002
            dof_acc = -1e-7
            # energy
            torques = 0.0
            dof_vel = 0.0
            energy = 0.0

            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class BruceWalkCfgCRPO(LeggedRobotCfgPPO):
    seed = 200
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner


    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]



    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        cost_critic_learning_rate = 1e-5

        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.95
        num_mini_batches = 4
        
        cost_gamma = 0.999
        cost_lam = 0.9
        use_cost_values = [False]
        cost_limit1 = BruceWalkCRPOCfg.env.cost_limit1
        cost_limit2 = BruceWalkCRPOCfg.env.cost_limit2
        
        lagrangian_multiplier_init1 = 0.0
        lambda_lr1 = 1e-3
        lambda_optimizer1 = "Adam"
        
        lagrangian_multiplier_init2 = 0.0
        lambda_lr2 = 1e-3
        lambda_optimizer2 = "Adam"
        
    class runner:
        policy_class_name = 'ActorCritic_RSLRL'

        lagrange_class_name = 'CRPOLagrange'
        algorithm_class_name = 'CRPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'exp'

        run_name = 'CRPO'
        # Load and resume
        resume = False
        load_run = 'crpo'
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_config = 'brucewalk_config_crpo.py'
