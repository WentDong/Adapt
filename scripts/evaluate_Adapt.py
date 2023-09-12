import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import yaml
from legged_gym.envs import *
from legged_gym.utils import (get_args, task_registry)
from model import Adapt

import isaacgym

import torch

codename_list = []	# 12 joints
for i in ["F", "R"]:
    for j in ["L", "R"]:
        for k in ["H", "T", "C"]:
            codename_list.append(j+i+k)

ENV_NUMS = 4096  # envs
 
def test_Adapt(args, env, model, faulty_tag = 0, flawed_rate = 1):
    # loading model
    eval_batch_size = ENV_NUMS  
    max_test_ep_len = 1000    	#iters

    body_dim = args["body_dim"]
    body_target = [1 for _ in range(body_dim)]
    body_target [faulty_tag] = flawed_rate
    
    state_mean = model.state_mean.clone().to(device)
    state_std = model.state_std.clone().to(device)
    
    body_target = torch.tensor(body_target, dtype=torch.float32, device=device)
    bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32)
    
    results = {}

    running_state = env.reset()[0]
    running_reward = torch.zeros((eval_batch_size, ), dtype=torch.float32, device=device)
    actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
    states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                        dtype=torch.float32, device=device)
    total_rewards = np.zeros(eval_batch_size)
    total_length = np.zeros(eval_batch_size)
    dones = np.zeros(eval_batch_size)
    
    #testing 
    with torch.no_grad():
        print(f"joint {codename_list[faulty_tag]} with degradation rate {flawed_rate} is under testing")
        for t in range(max_test_ep_len):

            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std

            if t < context_len:
                _, act_preds, _ = model.forward(
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            bodies=bodies[:,:context_len])
                act = act_preds[:, t].detach()
            else:
                _, act_preds, _ = model.forward(
                                        states[:,t-context_len+1:t+1],
                                        actions[:,t-context_len+1:t+1],
                                        bodies=bodies[:,t-context_len+1:t+1])
                act = act_preds[:, -1].detach()
            
            running_state, _, running_reward, done, infos, _, _= env.step(act, body_target)
            actions[:, t] = act

            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            total_length += (dones == 0)
            dones += done.detach().cpu().numpy()

            if np.all(dones):
                break
    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_length)
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")
    
    return results['eval/avg_reward'], results['eval/avg_ep_len']

if __name__ == '__main__':
    with open("./scripts/test_args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)

    device = torch.device(args["device"])  # setting flexible
    
    env_args = get_args()
    env_args.sim_device = args["device"]
    env_args.task = args["task"]
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = ENV_NUMS
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0, 1]
    env_cfg.commands.ranges.lin_vel_y = [-0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]
    # prepare environment
    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)
    
    # loading model
    # loading pre_record stds,means...
    log_dir = args["log_dir"]
    file_name = args["file_name"]
    model_path = os.path.join(parentdir, log_dir, file_name)
    task_args = {}
    with open(os.path.join(model_path, "args.yaml"), "r") as f:
        task_args_Loader = yaml.load_all(f, Loader = yaml.FullLoader)
        for t in task_args_Loader:
            task_args.update(t)
    state_mean, state_std = np.load(os.path.join(model_path, "model.state_mean.npy")), np.load(os.path.join(model_path, "model.state_std.npy"))
    
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args	["body_dim"]

    context_len = task_args["context_len"]      # K in decision transformer
    n_blocks = task_args['n_blocks']            # num of transformer blocks
    embed_dim = task_args['embed_dim']          # embedding (hidden) dim of transformer 
    n_heads = task_args['n_heads']              # num of transformer heads
    dropout_p = task_args['dropout_p']          # dropout probability
    position_encoding_length = task_args['position_encoding_length']
    device = torch.device(env_args.sim_device)
    model = Adapt(
            body_dim=body_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            state_mean=state_mean, 
            state_std=state_std,
            position_encoding_length=position_encoding_length
            ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_path,args["model_name"])
    , map_location=device))
    model.eval()
    #testing
    file_path = os.path.join(os.path.dirname(parentdir), "evals", log_dir, file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = "Returns.csv"
    length_name = "Lengths.csv"
    Adapt_Rows = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Adapt_table = np.zeros((12,10))
    len_table = np.zeros((12,10))
    for i in range(12):
        for j in Adapt_Rows:            
            Adapt_table[i, np.where(Adapt_Rows==j)], len_table[i, np.where(Adapt_Rows==j)] = test_Adapt(args, env, model, i, j)
            df = pd.DataFrame(Adapt_table)
            df.index = codename_list
            df.columns = Adapt_Rows
            df_res = df.to_csv(os.path.join(file_path, file_name), mode='w')

            df2 = pd.DataFrame(len_table)
            df2.index = codename_list
            df2.columns = Adapt_Rows
            df2_res = df2.to_csv(os.path.join(file_path, length_name), mode = 'w')
    