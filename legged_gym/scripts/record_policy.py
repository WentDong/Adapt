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

import cv2
import os

import numpy as np
from datetime import datetime

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from legged_gym import LEGGED_GYM_ROOT_DIR

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
from legged_gym.scripts.utils import flaw_generation, step_body

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # #Random
    # env_cfg.commands.ranges.lin_vel_x = [-0.7, 0.7]
    # env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]
    # video_file_name = "Random"


    # Forward
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0.7]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    video_file_name = "Forward"


    # #Backward
    # env_cfg.commands.ranges.lin_vel_x = [-0.7, -0.7]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    # video_file_name = "Backward"

    
    # # Lateral
    # env_cfg.commands.ranges.lin_vel_x = [0, 0]
    # env_cfg.commands.ranges.lin_vel_y = [0.5, 0.5]
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    # video_file_name = "Lateral"

    # # Rotate
    # env_cfg.commands.ranges.lin_vel_x = [0, 0]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.ang_vel_yaw = [1, 1]
    # video_file_name = "Rotate"
    
    # # Circle
    # env_cfg.commands.ranges.lin_vel_x = [0.7, 0.7]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.ang_vel_yaw = [1, 1]
    # video_file_name = "Circle"
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    img_idx = 0

    video_duration = 15
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    joint = args.joint
    rate = args.rate
    body_latency = args.body_latency
    video_file_name += "body_latency" + str(body_latency)
    # video_file_name += "Always_Given_Body1"
    bodies, joints = flaw_generation(obs.shape[0], fixed_joint=joint, flawed_rate=rate, device = env.device)
    origin_bodies = bodies.clone()
    body_buffer = [bodies] * (body_latency+1)
    length_body_buffer = body_latency + 1
    cur_body_index = 0

    if not os.path.exists(video_file_name):
        os.mkdir(video_file_name)

    video_name = f"record_Joint{joint}.mp4"
    for i in range(num_frames):
        # actions = policy(obs.detach(), body_buffer[cur_body_index].detach())
        actions = policy(obs.detach(), origin_bodies.detach()) 
        # import pdb; pdb.set_trace()

        obs, _, _, _, infos, _, _ = env.step(actions.detach(), bodies)
        if i==200:
            bodies[:, joint] = 0
        # bodies = step_body(bodies, joints, 0.05, 0.0001)
       
        # Reset camera position.
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
        camera_relative_position = 1.2 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
        env.set_camera(look_at + camera_relative_position, look_at)
        # import pdb; pdb.set_trace()
        if RECORD_FRAMES:
            frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
            if not os.path.isdir(frames_path):
                os.mkdir(frames_path)
            filename = os.path.join('logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(os.path.join(video_file_name, video_name), cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 

    video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    args = get_args()
    play(args)