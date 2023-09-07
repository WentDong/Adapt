"""
Test the performance of teacher policy.
"""

import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
from collections import deque

import numpy as np
import pandas as pd
import yaml
from legged_gym.envs import *
from legged_gym.utils import (Logger, export_policy_as_jit, get_args,
                              task_registry)
from legged_gym.scripts.utils import flaw_generation

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR

import torch

codename_list = []
for i in ["F", "R"]:
    for j in ["L", "R"]:
        for k in ["H", "T", "C"]:
            codename_list.append(i+j+k)

ENV_NUMS = 4096  #测试环境数
def test_ppo(args, env, train_cfg, faulty_tag = -1, flawed_rate = 1):
    """在单次循环中

    Args:
        args (_type_): 就各种参数
        env ( optional): 用来测试的环境
        train_cfg ( optional): 用来训练脚本
        faulty_tag (int, optional): 坏的关节 -1为全好. Defaults to -1.
        flawed_rate (int, optional): 坏的成都  1为完好. Defaults to 1.
    """
    
    obs = env.reset()[0]

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.run_name
    train_cfg.runner.experiment_name = args.experiment_name
    train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, evaluate=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    body, _ = flaw_generation(env.num_envs, bodydim=12, fixed_joint=[faulty_tag], flawed_rate=flawed_rate, device=env.device)
    dones = np.zeros(env_cfg.env.num_envs)

    total_rewards = np.zeros(env_cfg.env.num_envs)
    total_length = np.zeros(env_cfg.env.num_envs)
    for i in range(int(env.max_episode_length)-1):
        actions = policy(obs.detach(), body)
        obs, privileged_obs, rews, done, infos, reset_env_ids, terminal_amp_states = env.step(actions.detach(), body)
        total_rewards += rews.detach().cpu().numpy() * (dones == 0)
        total_length += (dones == 0)
        dones += done.detach().cpu().numpy()

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_length)
    print(f"average reward is :{avg_reward}\naverage length is :{avg_length}\n")
    return avg_reward, avg_length
            
if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = ENV_NUMS
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0]
    env_cfg.terrain.measure_heights = False
    env_cfg.commands.ranges.lin_vel_x = [0, 1]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]
    env_cfg.commands.curriculum=False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    ppo_row_names = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out_table = np.zeros((12,10))
    out_table2 = np.zeros((12,10))
    save_path = os.path.join(os.path.dirname(parentdir), "eval")
    file_name = args.file_name + "_returns.xlsx"
    len_name = args.file_name + "_lengths.xlsx"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.joint==-1:
        for i in range(12):
            for t, j in enumerate(ppo_row_names):
                out_table[i, t], out_table2[i, t] = test_ppo(args, env, train_cfg, i, j)
                ppo_df = pd.DataFrame(out_table)
                ppo_df.index = codename_list
                ppo_df.columns = ppo_row_names
                ppo_res = ppo_df.to_excel(os.path.join(save_path, file_name))

                df2 = pd.DataFrame(out_table2)
                df2.index = codename_list
                df2.columns = ppo_row_names
                df2_res = df2.to_excel(os.path.join(save_path, len_name))
    else:
        i = args.joint
        for t, j in enumerate(ppo_row_names):
            out_table[i, t], out_table2[i, t] = test_ppo(args, env, train_cfg, i, j)
            ppo_df = pd.DataFrame(out_table)
            ppo_df.index = codename_list
            ppo_df.columns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ppo_res = ppo_df.to_excel(os.path.join(save_path, file_name))

            df2 = pd.DataFrame(out_table2)
            df2.index = codename_list
            df2.columns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            df2_res = df2.to_excel(os.path.join(save_path, len_name))
