import argparse
from argparse import Namespace
import os
import pickle
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from datetime import datetime

import numpy as np

from legged_gym.envs import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from model import LeggedTransformerPro, LeggedTransformerBody
from legged_gym.utils import  get_args, task_registry, Logger
import pdb
from collections import Counter
import yaml

from tqdm import trange, tqdm


def play(args, faulty_tag = -1, flawed_rate = 1):
    
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args["body_dim"]
    ##-----

    context_len = args["context_len"]      # K in decision transformer
    n_blocks = args["n_blocks"]            # num of transformer blocks
    embed_dim = args["embed_dim"]          # embedding (hidden) dim of transformer
    n_heads = args["n_heads"]              # num of transformer heads
    dropout_p = args["dropout_p"]          # dropout probability
    device = torch.device(args["device"])
    

    print("loading pre_record stds,means...")
    model_path = os.path.join(parentdir, "EAT_runs_AMP/EAT_NEW_URDF_NEWURDFSTEP2_03/")
    # model_path = os.path.join(parentdir, "EAT_runs/EAT_FLAWEDPPO_00/")
    state_mean, state_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy")
    # state_mean, state_std, body_mean, body_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy"), np.load(model_path+"model.body_mean.npy"), np.load(model_path+"model.body_std.npy")
    #init eval para
    eval_batch_size = 50  # envs
    max_test_ep_len=1000    #iters
    nobody = False
    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = 48
    act_dim = 12
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)
    
    #======================================================================
    #prepare envs
    env_args = get_args()
    env_args.sim_device = args["device"]
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args["task"])
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, eval_batch_size)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]# 更改速度设置以防命令采样到0的情况    

    # env = A1(num_envs=args.num_eval_ep, noise=args.noise)     #另一种环境初始化方式
    env, _ = task_registry.make_env(name=args["task"], args=env_args, env_cfg=env_cfg)
    obs = env.get_observations()
    #===================================================================================
    
    
    #====================================================================================
    # prepare algs
    model = LeggedTransformerBody(#! 原为LeggedTransformerPro，发现改为Body后狗直接不走
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            # state_mean=state_mean,
            # state_std=state_std,
            # body_mean=body_mean,
            # body_std=body_std
            ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_path,"model2600epoch.pt"), map_location = "cuda:0"
    ))
    model.eval()
    #====================================================================================
    
        
    #====================================================================================
    ##eval pre
    #init visualize
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    logger = Logger(env.dt)
    
    with torch.no_grad():
        if True:
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            body_oringe = [1 for _ in range(12)]
            body_dim = len(body_oringe)
            
            # if (faulty_tag != -1):
            #     body_oringe[faulty_tag] = flawed_rate
            # body_target = torch.tensor(body_oringe, dtype=torch.float32, device=device)
            # body_target = (torch.tensor(body_target, dtype=torch.float32, device=device) - body_mean) / body_std
            # bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()

            # init episode
            running_state = env.reset()[0]
            running_reward = torch.zeros((eval_batch_size, ),
                                dtype=torch.float32, device=device)

            total_rewards = np.zeros(eval_batch_size)
            dones = np.zeros(eval_batch_size)

            for t in range(max_test_ep_len):

                total_timesteps += (dones == 0)
                faulty_taget = faulty_tag
                body_target = body_oringe.copy()
                body_target[faulty_taget] = flawed_rate
                body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
                bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()

                states[:,t,:] = running_state
                states[:,t,:] = (states[:,t,:] - state_mean) / state_std

                if t < context_len:
                    _, act_preds, body_preds = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                body=bodies[:,:context_len])
                    act = act_preds[:, t].detach()
                    # body = body_preds[:, t].detach()
                else:
                    _, act_preds, body_preds = model.forward(timesteps[:,t-context_len+1:t+1],
                                            states[:,t-context_len+1:t+1],
                                            actions[:,t-context_len+1:t+1],
                                            # body=bodies[:,t-context_len+1:t+1])
                                            body=bodies[:,t-context_len+1:t+1] if not nobody else None)
                    act = act_preds[:, -1].detach()
                    bodies[:, t] = body_preds[:, -1].detach()   #加这一句可以让学出来的body返回回去
                
                running_state, _, running_reward, done, infos = env.step(act, flawed_joint = [faulty_taget], flawed_rate = flawed_rate) #if t > max_test_ep_len/8 else env.step(act, [-1])
                
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
                    
                # if RECORD_FRAMES:
                #     if t % 2:
                #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                #         env.gym.write_viewer_image_to_file(env.viewer, filename) 
                #         img_idx += 1 
                # if MOVE_CAMERA: #TODO: 这里可以设定视角变换，后续学习一下
                #     camera_position += camera_vel * env.dt
                #     env.set_camera(camera_position, camera_position + camera_direction)
            
                actions[:, t] = act

                total_reward += np.sum(running_reward.detach().cpu().numpy())
                total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
                dones += done.detach().cpu().numpy()

                if torch.all(done):
                    break

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_timesteps)
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")
    logger.print_rewards()

def play_withbody(args, faulty_tag = -1, flawed_rate = 1):
    # rtg_scale = 1000      # normalize returns to go
    # state_dim = 48
    # act_dim = 12
    # body_dim = 12

    # context_len = 20      # K in decision transformer
    # n_blocks = 6            # num of transformer blocks
    # embed_dim = 128          # embedding (hidden) dim of transformer #! 原值128 #512
    # n_heads = 1              # num of transformer heads
    # dropout_p = 0.1          # dropout probability

    print("loading pre_record stds,means...")
    model_name = "EAT_Small_Damp_SMALLDAMPNEWREWARD_03"
    # model_name = "EAT_Mix_DAMP_MIXDAMP_00"
    run_name = "EAT_runs_AMP"
    model_path = os.path.join(os.path.dirname(currentdir), run_name, model_name)


    task_args = {}
    with open(os.path.join(model_path, "args.yaml"), "r") as f:
        task_args_Loader = yaml.load_all(f, Loader = yaml.FullLoader)
        for t in task_args_Loader:
            task_args.update(t)


    # model_path = os.path.join(parentdir, "EAT_runs/EAT_FLAWEDPPO_00/")
    state_mean, state_std= np.load(os.path.join(model_path, "model.state_mean.npy")), np.load(os.path.join(model_path, "model.state_std.npy"))
    # state_mean, state_std, body_mean, body_std = np.load(model_path+"model.state_mean.npy"), np.load(model_path+"model.state_std.npy"), np.load(model_path+"model.body_mean.npy"), np.load(model_path+"model.body_std.npy")
    body_mean, body_std = None, None
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args	["body_dim"]

    position_encoding_length = task_args["position_encoding_length"]
    context_len = task_args["context_len"]      # K in decision transformer
    n_blocks = task_args['n_blocks']            # num of transformer blocks
    embed_dim = task_args['embed_dim']          # embedding (hidden) dim of transformer 
    n_heads = task_args['n_heads']              # num of transformer heads
    dropout_p = task_args['dropout_p']          # dropout probability
    pred_body = task_args.get('pred_body', True)
    #======================================================================
    #prepare envs
    env_args = get_args()
    env_args.sim_device = args["device"]
    env_args.task = args["task"]
    eval_batch_size = 10  # envs
    max_test_ep_len=1001    #iters
    nobody = False
    results = {}
    total_reward = 0
    total_timesteps = 0

    env_cfg, _ = task_registry.get_cfgs(name =env_args.task)
    env_cfg.env.num_envs = eval_batch_size
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.7]# 更改速度设置以防命令采样到0的情况    
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.sim_device = args["device"]
    env_cfg.commands.ranges.lin_vel_x = [0.4, 0.4]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env, _ = task_registry.make_env(name = env_args.task, args = env_args, env_cfg = env_cfg)
    device = torch.device(env_args.sim_device)
    
    # obs = env.get_observations()
    #===================================================================================
    
    #====================================================================================
    # prepare algs
    # device = torch.device(args["device"])
    model = LeggedTransformerBody(
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            position_encoding_length=position_encoding_length,
            ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_path,"model_best.pt"), map_location = args["device"]
        # os.path.join(model_path, "model4000epoch.pt"), map_location=args["device"]
    ))
    model.eval()
    #====================================================================================
    
        
    #====================================================================================
    ##eval pre
    #init visualize
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    #init eval para

    state_dim = 48
    act_dim = 12
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)
    # body_mean = torch.from_numpy(body_mean).to(device)
    # body_std = torch.from_numpy(body_std).to(device)
    
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    logger = Logger(env.dt)
    
    with torch.no_grad():
        if True:
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)

            body_oringe = torch.ones(12)
            body_dim = 12 
            
            # if (faulty_tag != -1):
            #     body_oringe[faulty_tag] = flawed_rate
            # body_target = torch.tensor(body_oringe, dtype=torch.float32, device=device)
            # body_target = (torch.tensor(body_target, dtype=torch.float32, device=device) - body_mean) / body_std
            # bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()

            # init episode
            running_state = env.reset()[0]
            running_reward = torch.zeros((eval_batch_size, ),
                                dtype=torch.float32, device=device)

            total_rewards = np.zeros(eval_batch_size)
            dones = np.zeros(eval_batch_size)
            
            faulty_taget = faulty_tag
            body_target = body_oringe.clone()
            # body_target[faulty_taget] = flawed_rate
            body_target = body_target.to(device)
            # body_target = r(body_target, dtype=torch.float32, device=device)
            # faulty_taget = -1
            running_body = body_target.expand(eval_batch_size, body_dim).type(torch.float32).clone()
            bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()
            
            # change_timing = torch.randint(110,250,(eval_batch_size,))
            change_timing = torch.zeros(eval_batch_size)
            for t in range(max_test_ep_len):
                if MOVE_CAMERA:
                    lootat = env.root_states[9, :3]
                    # camara_position = lootat.detach().cpu().numpy() + [0, 1, 0.5]
                    camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
                    # camara_position = lootat.detach().cpu().numpy() + [-1, 0, 0]
                    env.set_camera(camara_position, lootat)
                total_timesteps += (dones == 0)

                states[:,t,:] = running_state
                states[:,t,:] = (states[:,t,:] - state_mean) / state_std
                bodies[:,t] = running_body
                if t < context_len:
                    _, act_preds, body_preds = model.forward(
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                bodies=bodies[:,:context_len])
                    act = act_preds[:, t].detach()
                    # faulty_taget = -1
                    # body = body_preds[:, t].detach()
                else:
                    if pred_body:
                        _, _, body_preds = model.forward(
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                bodies=bodies[:,t-context_len+1:t+1])
                        # if t < 300:
                        #     bodies[:, t] = torch.ones_like(body_preds[:, -1], dtype=torch.float64)
                        # else:
                        bodies[:, t] = body_preds[:, -1].detach()   #加这一句可以让学出来的body返回回去
                    # faulty_taget = faulty_tag
                    # if t % 900 == 0 and t > 0:
                    #     pdb.set_trace()
                    # print(torch.argmin(bodies[:,t,:], dim=-1))
                    _, act_preds, body_preds = model.forward(
                                            states[:,t-context_len+1:t+1],
                                            actions[:,t-context_len+1:t+1],
                                            bodies=bodies[:,t-context_len+1:t+1])
                    act = act_preds[:, -1].detach()

                #let one joint or leg be disabled
                # act = disable_leg(act.detach(), target="none", index=3)
                # running_state, running_reward, done, _ = env.step(act.cpu())
                for j in range(eval_batch_size):
                    if change_timing[j] == t:
                        running_body[j][faulty_taget] = flawed_rate
                running_state, _, running_reward, done, infos = env.step(act, running_body) #if t > max_test_ep_len/8 else env.step(act, [-1])
                


                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
                    
                # if RECORD_FRAMES:
                #     if t % 2:
                #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                #         env.gym.write_viewer_image_to_file(env.viewer, filename) 
                #         img_idx += 1 
                # if MOVE_CAMERA: #TODO: 这里可以设定视角变换，后续学习一下
                #     camera_position += camera_vel * env.dt
                #     env.set_camera(camera_position, camera_position + camera_direction)
            
                actions[:, t] = act

                total_reward += np.sum(running_reward.detach().cpu().numpy())
                total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
                dones += done.detach().cpu().numpy()

                if torch.all(done):
                    break

    # results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_timesteps)
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")
    logger.print_rewards()


if __name__ == "__main__":
    with open("./Integration_EAT/scripts/test_args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)
    
    RECORD_FRAMES = False
    # MOVE_CAMERA = False
    MOVE_CAMERA = True
    # play(args, 0, 0)
    play_withbody(args, 2, 0)
