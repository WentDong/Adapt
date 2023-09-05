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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class A1AMPCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 48 #+ 11 * 17
        num_privileged_obs = 48 #+ 11 * 17
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            #raw pos
            # 'FL_hip_joint': 0.0,  # [rad]
            # 'RL_hip_joint': 0.0,  # [rad]
            # 'FR_hip_joint': 0.0,  # [rad]
            # 'RR_hip_joint': 0.0,  # [rad]
            #
            # 'FL_thigh_joint': 0.9,  # [rad]
            # 'RL_thigh_joint': 0.9,  # [rad]
            # 'FR_thigh_joint': 0.9,  # [rad]
            # 'RR_thigh_joint': 0.9,  # [rad]
            #
            # 'FL_calf_joint': -1.8,  # [rad]
            # 'RL_calf_joint': -1.8,  # [rad]
            # 'FR_calf_joint': -1.8,  # [rad]
            # 'RR_calf_joint': -1.8,  # [rad]
            #pos2
            # 'FL_hip_joint': 0.0,   # [rad]
            # 'RL_hip_joint': 0.0,   # [rad]
            # 'FR_hip_joint': 0.0 ,  # [rad]
            # 'RR_hip_joint': 0.0,   # [rad]
            #
            # 'FL_thigh_joint': 0.8,     # [rad]
            # 'RL_thigh_joint': 0.8,   # [rad]
            # 'FR_thigh_joint': 0.8,     # [rad]
            # 'RR_thigh_joint': 0.8,   # [rad]
            #
            # 'FL_calf_joint': -1.5,   # [rad]
            # 'RL_calf_joint': -1.5,    # [rad]
            # 'FR_calf_joint': -1.5,  # [rad]
            # 'RR_calf_joint': -1.5,    # [rad]
            #pos3
            # 'FL_hip_joint': 0.0,  # [rad]
            # 'RL_hip_joint': 0.0,  # [rad]
            # 'FR_hip_joint': 0.0,  # [rad]
            # 'RR_hip_joint': 0.0,  # [rad]
            #
            # 'FL_thigh_joint': 0.7,  # [rad]
            # 'RL_thigh_joint': 0.7,  # [rad]
            # 'FR_thigh_joint': 0.7,  # [rad]
            # 'RR_thigh_joint': 0.7,  # [rad]
            #
            # 'FL_calf_joint': -1.4,  # [rad]
            # 'RL_calf_joint': -1.4,  # [rad]
            # 'FR_calf_joint': -1.4,  # [rad]
            # 'RR_calf_joint': -1.4,  # [rad]
            #pos4
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        # penalize_contacts_on = ["thigh", "calf", "base"]
        # terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.05, 2.75]
        randomize_restitution = True
        restitution_range = [0, 1]

        randomize_base_mass = True
        added_mass_range = [-1., 2.]  #kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True

        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = False
        action_latency_range = [0, 3] #*dt = 0.005

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            # dof_pos = 0.03
            # dof_vel = 1.5
            # lin_vel = 0.1
            # ang_vel = 0.3
            # gravity = 0.05
            # height_measurements = 0.1

            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            # termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05 #0#-0.05
            orientation = -2 #-0.005 #-2.0
            torques = -0.0001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0 # 1.0
            collision = -0.5 #-0.1
            feet_stumble = -0.0
            action_rate = -0.01
            # dof_pos_dif = -0.1
            # stand_still = -0.
            action_magnitude = -0.3

            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = 0#-2.0
            # ang_vel_xy = 0#-0.05
            # orientation = 0#-2.0
            # torques = -0.0001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time =  1.0
            # collision = -0.1
            # feet_stumble = -0.0
            # action_rate = -0.01
            # action_magnitude = 0#-0.3
            # dof_pos_dif = 0#-0.1
            # # stand_still = -0.
            # smoothness = 0#-0.01
            # power = 0#-5.0e-4
            # power_distribution = 0

    class commands:
        # curriculum = False
        # max_curriculum = 1.
        curriculum =  True
        min_lin_vel_x_curriculum = -0.0
        max_lin_vel_x_curriculum = 1
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 0.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.05, 0.4] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            heading = [-3.14/4, 3.14/4]

class A1AMPCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'terrain_small_velrange'
        experiment_name = 'a1_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000 # number of policy updates

        # amp_reward_coef = 2.0
        amp_reward_coef = 0.5 * 0.02
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  