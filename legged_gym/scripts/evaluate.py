"""
本脚本用于自动测试某个模型在各种坏损程度下的得分
"""

import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
import statistics
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

codename_list = []	#存储每条腿的字母代号
for i in ["F", "R"]:
    for j in ["L", "R"]:
        for k in ["H", "T", "C"]:
            codename_list.append(i+j+k)

ENV_NUMS = 4096  #测试环境数
FASTER = True
def test_ppo(args, env, train_cfg, faulty_tag = -1, flawed_rate = 1, stuck = False):
    """在单次循环中

    Args:
        args (_type_): 就各种参数
        env ( optional): 用来测试的环境
        train_cfg ( optional): 用来训练脚本
        faulty_tag (int, optional): 坏的关节 -1为全好. Defaults to -1.
        flawed_rate (int, optional): 坏的成都  1为完好. Defaults to 1.
    """
    
    obs = env.reset()[0]
    # obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # train_cfg.runner.load_run = "strange"
    # if faulty_tag == -1:
    #     train_cfg.runner.load_run = f""
    # else:
    #     train_cfg.runner.load_run = f"{faulty_tag}"
    train_cfg.runner.load_run = "Small_Damp_No_Body"
    # train_cfg.runner.checkpoint = faulty_tag
    train_cfg.runner.experiment_name = "Models"
    # train_cfg.runner.load_run = "Aug04_08-07-47_PPO_ALL"
    train_cfg.runner.checkpoint = 1111
    # train_cfg.runner.experiment_name = "stuck"
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    #判断模型文件是否存在 若不存在则报错弹出
    # if not os.path.exists(os.path.join(log_root,train_cfg.runner.load_run)):
    #     print(f"no model file{faulty_tag}_{flawed_rate}")
    #     return -1 , 0

    # train_cfg.runner.checkpoint = -1    #! 改成best
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, evaluate=True)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    cur_reward_sum = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    rewbuffer = deque(maxlen=100)
    cur_episode_length = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.sim_device)
    lengthbuffer = deque(maxlen=100)
    
    body, _ = flaw_generation(env.num_envs, bodydim=12, fixed_joint=[faulty_tag], flawed_rate=flawed_rate, device=env.device)
    fake_body = torch.ones((env.num_envs, 12), dtype=torch.float, device = args.sim_device)
    print(body[:3], body.shape)

    dones = np.zeros(env_cfg.env.num_envs)

    total_rewards = np.zeros(env_cfg.env.num_envs)
    total_length = np.zeros(env_cfg.env.num_envs)
    for i in range(int(env.max_episode_length)-1):
        actions = policy(obs.detach(), fake_body)
        # actions = policy(obs.detach(), body)
        obs, privileged_obs, rews, done, infos, reset_env_ids, terminal_amp_states = env.step(actions.detach(), body, stuck)
        # next_amp_obs = env.get_amp_observations()
        # next_amp_obs_with_term = torch.clone(next_amp_obs)
        # next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
        total_rewards += rews.detach().cpu().numpy() * (dones == 0)
        total_length += (dones == 0)
        dones += done.detach().cpu().numpy()
        
        # cur_reward_sum += rews
        # new_ids = (dones > 0).nonzero(as_tuple=False)
        # rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        # cur_reward_sum[new_ids] = 0
        # cur_episode_length += torch.ones(env_cfg.env.num_envs,dtype=torch.float, device=args.sim_device)
        # lengthbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        # cur_episode_length[new_ids] = 0
        # dones += done
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i % stop_rew_log == 0 and i != 0:
        #     logger.print_rewards()
        #     if len(rewbuffer)>0:
        #         print(f"average reward is :{statistics.mean(rewbuffer)}\naverage length is :{statistics.mean(lengthbuffer)}\n")
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_length)
    print(f"average reward is :{avg_reward}\naverage length is :{avg_length}\n")
    # return statistics.mean(rewbuffer), statistics.mean(lengthbuffer)
    return avg_reward, avg_length
            
if __name__ == '__main__':
    # with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
    #     args = yaml.safe_load(fargs)

    # device = torch.device(args["device"])  # setting flexible
    
    args = get_args()
    # env_args.sim_device = args["device"]
    args.rl_device = args.sim_device
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = ENV_NUMS
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]# 更改速度设置以防命令采样到0的情况    

    # env_cfg.domain_rand.randomize_action_latency = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.randomize_gains = False
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    # env_cfg.domain_rand.randomize_motor_strength = False
    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0]
    env_cfg.terrain.measure_heights = False
    env_cfg.commands.ranges.lin_vel_x = [0, 1]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]
    # train_cfg.runner.amp_reward_coef=0
    env_cfg.commands.curriculum=False
    stuck = args.stuck
    #Faster settings:
    # if FASTER:
    #     env_cfg.commands.ranges.lin_vel_x = [-0.7, 0.7]
    #     env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    #     env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    #测试ppo======================================================================
    # ppo_row_names = [0, 0.25, 0.5, 0.75]
    ppo_row_names = np.arange(0,1,0.1)
    # ppo_row_names = np.arange(0.9,1,0.1)
    out_table = np.zeros((12,10))
    out_table2 = np.zeros((12,10))
    save_path = os.path.join(os.path.dirname(parentdir), "eval")
    file_name = "Small_Damp_Old_Reward_G_No_Body.xlsx"
    len_name = "Small_Damp_Old_Reward_G_No_Body_len.xlsx"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(12):#12条断腿情况
    # for i in [2,5]:
        # for j in range (0, 0.8, 0.1):
        t = 0
        for j in ppo_row_names:            
            out_table[i, t], out_table2[i, t] = test_ppo(args, env, train_cfg, i, j, stuck)
            t += 1
            ppo_df = pd.DataFrame(out_table)
            ppo_df.index = codename_list
            ppo_df.columns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ppo_res = ppo_df.to_excel(os.path.join(save_path, file_name))

            df2 = pd.DataFrame(out_table2)
            df2.index = codename_list
            df2.columns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            df2_res = df2.to_excel(os.path.join(save_path, len_name))


    # out_table[:,-1],_ = test_ppo(args, env, train_cfg, -1, 1) #测完好情况
    # ppo_df = pd.DataFrame(out_table)
    # ppo_df.index = codename_list
    # # ppo_df.columns = [0,0.25,0.5, 0.75, 1.0]
    # ppo_df.columns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ppo_res = ppo_df.to_csv(os.path.join(save_path, file_name), mode='w')
   