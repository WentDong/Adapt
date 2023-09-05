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


import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from legged_gym.scripts.utils import flaw_generation, step_body


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    # env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False

    # env_cfg.domain_rand.randomize_action_latency = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.randomize_gains = False
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    # env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1


    # env_cfg.terrain.terrain_proportions = [0.0, 0, 0, 0,0, 1.0]

    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0]



    # env_cfg.terrain.terrain_proportions = [0.0, 0, 0, 0, 0, 1.0]
    # env_cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0, 0]
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0]
    # env_cfg.terrain.terrain_proportions = [0, 0, 0.0, 0, 1.0]
    # env_cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0]
    env_cfg.commands.ranges.lin_vel_x = [1, 1]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.ranges.heading = [0., 0]
    # env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.ang_vel_yaw = [1, 1]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # train_cfg.runner.load_run = 'Jul02_19-10-19_plane_collect_rate_rldata'
    # train_cfg.runner.load_run = 'Jul01_16-56-15_plane_collect'
    # train_cfg.runner.load_run = 'Jul02_15-58-46_plane_collect_rate'
    # train_cfg.runner.load_run = 'Jul05_22-35-18_latency'
    # train_cfg.runner.checkpoint = 10000


    joint = args.joint
    rate = args.rate
    body_latency = args.body_latency
    stuck = args.stuck

    bodies, joints = flaw_generation(obs.shape[0], fixed_joint=joint, flawed_rate=1, device = env.device)
    fake_bodies = bodies.clone()
    body_buffer = [bodies] * (body_latency+1)
    length_body_buffer = body_latency + 1
    cur_body_index = 0

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, evaluate=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # change_point = torch.randint(int(env.max_episode_length)//10, int(env.max_episode_length)//5, (obs.shape[0],), device=env.device)
    change_point = torch.zeros(obs.shape[0], device = env.device)
    record_actions = []
    lootat = env.root_states[9, :3]
    # camara_position = lootat.detach().cpu().numpy() + [0, 1, 0.5]
    camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
    # camara_position = lootat.detach().cpu().numpy() + [-1, 0, 0]
    env.set_camera(camara_position, lootat)
    for i in range(int(env.max_episode_length)+1):
        # actions = policy(obs.detach(), body_buffer[cur_body_index].detach())
        actions = policy(obs.detach(), fake_bodies.detach())
        # actions[:] = 0

        record_actions.append(actions)

        obs, _, rews, dones, infos, _, _ = env.step(actions.detach(), bodies, stuck)
        for j in range(obs.shape[0]):
            if change_point[j] == i:
                bodies[j, joint] = rate
        # bodies = step_body(bodies, joints, 0.2, 0.001, 1)
        body_buffer[cur_body_index] = bodies

        cur_body_index += 1
        if cur_body_index == length_body_buffer:
            cur_body_index = 0
        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[9, :3]
            # camara_position = lootat.detach().cpu().numpy() + [0, 1, 0.5]
            camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
            # camara_position = lootat.detach().cpu().numpy() + [-1, 0, 0]
            env.set_camera(camara_position, lootat)
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # elif i==stop_state_log:
        #     logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    logger.print_rewards()

    record_actions = torch.concat(record_actions,dim=0)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    # MOVE_CAMERA = False
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
