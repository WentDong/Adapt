import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np

from legged_gym.envs import *

import torch
from model import Adapt
from legged_gym.utils import get_args, task_registry, Logger
import yaml

def play(args):

    print("loading pre_record stds,means...")
    file_name = args["file_name"]
    log_dir = args["log_dir"]
    model_path = os.path.join(os.path.dirname(currentdir), log_dir, file_name)


    task_args = {}
    with open(os.path.join(model_path, "args.yaml"), "r") as f:
        task_args_Loader = yaml.load_all(f, Loader = yaml.FullLoader)
        for t in task_args_Loader:
            task_args.update(t)


    state_mean, state_std= np.load(os.path.join(model_path, "model.state_mean.npy")), np.load(os.path.join(model_path, "model.state_std.npy"))
    state_dim = args["state_dim"]
    act_dim = args["act_dim"]
    body_dim = args	["body_dim"]
    joint = args["joint"]
    rate = args["rate"]

    position_encoding_length = task_args["position_encoding_length"]
    context_len = task_args["context_len"]      # K in decision transformer
    n_blocks = task_args['n_blocks']            # num of transformer blocks
    embed_dim = task_args['embed_dim']          # embedding (hidden) dim of transformer 
    n_heads = task_args['n_heads']              # num of transformer heads
    dropout_p = task_args['dropout_p']          # dropout probability
    #======================================================================
    #prepare envs
    env_args = get_args()
    env_args.sim_device = args["device"]
    env_args.task = args["task"]
    eval_batch_size = 10  # envs
    max_test_ep_len = 1001    #iters
    results = {}
    total_reward = 0
    total_timesteps = 0

    env_cfg, _ = task_registry.get_cfgs(name =env_args.task)
    env_cfg.env.num_envs = eval_batch_size
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.sim_device = args["device"]
    env_cfg.commands.ranges.lin_vel_x = [0, 1]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]
    env, _ = task_registry.make_env(name = env_args.task, args = env_args, env_cfg = env_cfg)
    device = torch.device(env_args.sim_device)
    
    #===================================================================================
    model = Adapt(
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
        os.path.join(model_path, args["model_name"]), map_location = args["device"]
    ))
    model.eval()
    #====================================================================================
    
        
    #====================================================================================
    #init eval para

    state_dim = 48
    act_dim = 12
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    logger = Logger(env.dt)
    
    with torch.no_grad():
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=torch.float32, device=device)

        # init episode
        running_state = env.reset()[0]
        running_reward = torch.zeros((eval_batch_size, ),
                            dtype=torch.float32, device=device)

        total_rewards = np.zeros(eval_batch_size)
        dones = np.zeros(eval_batch_size)
        
        body_target = torch.ones(body_dim).clone()
        body_target = body_target.to(device)

        running_body = body_target.expand(eval_batch_size, body_dim).type(torch.float32).clone()
        bodies = body_target.expand(eval_batch_size, max_test_ep_len, body_dim).type(torch.float32).clone()
        
        change_timing = torch.randint(110,250,(eval_batch_size,))
        for t in range(max_test_ep_len):
            if MOVE_CAMERA:
                lootat = env.root_states[9, :3]
                camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
                env.set_camera(camara_position, lootat)
            total_timesteps += (dones == 0)

            states[:,t,:] = running_state
            states[:,t,:] = (states[:,t,:] - state_mean) / state_std
            bodies[:,t] = running_body
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

            for j in range(eval_batch_size):
                if change_timing[j] == t:
                    running_body[j][joint] = rate
            running_state, _, running_reward, done, infos, _, _ = env.step(act, running_body)
            
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
                
            actions[:, t] = act

            total_reward += np.sum(running_reward.detach().cpu().numpy())
            total_rewards += running_reward.detach().cpu().numpy() * (dones == 0)
            dones += done.detach().cpu().numpy()

    results['eval/avg_reward'] = np.mean(total_rewards)
    results['eval/avg_ep_len'] = np.mean(total_timesteps)
    print(f"average reward is :{results['eval/avg_reward']}\naverage length is :{results['eval/avg_ep_len']}\n")
    logger.print_rewards()


if __name__ == "__main__":
    with open("./scripts/test_args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)
    
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    play(args)
