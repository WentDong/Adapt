import glob
import multiprocessing
import os
import pdb
import random
import time
import pickle
from regex import F
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum

def partial_traj(dataset_path_list, args, context_len=20, rtg_scale=1000, body_dim=12):
    """
    当轨迹过长的时候，为照顾cpu运算负荷需要将数据集分几段加载，此函数处理一段，可以通过简单的复用调用来完成
    -------------
    输入：
    dataset_path_list: list of trajs
        一组轨迹的存储路径
    batch_size: int
        training batch size
    context_len: int
        transformer需要读取的上下文长度
    rtg_scale: int
        normalize returns to go
    body_dim: int
        number of body parts


    输出：
    traj_data_loader：DataLoader objects
        用以学习的轨迹对象
    state_mean,state_std: float
        状态值的均值和方差
    body_mean,body_std: float
        body值的均值方差
    """
    big_list = []
    for pkl in tqdm(dataset_path_list):
        with open(pkl, "rb") as f:
            thelist = pickle.load(f)

        assert "body" in thelist[0]
        if args.cut == 0:
            big_list = big_list + thelist
        else:
            big_list = big_list + thelist[: args.cut]

    traj_dataset = D4RLTrajectoryDataset(
        big_list, context_len, rtg_scale, leg_trans_pro=True
    )
    assert body_dim == traj_dataset.body_dim

    state_mean, state_std = traj_dataset.get_state_stats(body=False)

    return traj_dataset, state_mean, state_std  # , body_mean, body_std

def flaw_generation(
    num_envs, bodydim=12, fixed_joint=[-1], flawed_rate=-1, device="cpu", upper_bound=1
):
    """
    num_envs: 环境数
    fixed_joint: 指定损坏的关节为fixed_joint(LIST) [0,11]，若不固定为-1
    flawed_rate: 损坏程度为flawed_rate, 若随机坏损为-1
    t(num_envs * len(fixed_joint)): 坏损的关节
    """
    if bodydim == 0:
        return None, None
    t = torch.randint(0, bodydim, (num_envs, 1))
    if -1 not in fixed_joint:
        t = torch.ones((num_envs, len(fixed_joint)), dtype=int) * torch.tensor(
            fixed_joint
        )
    bodies = torch.ones(num_envs, bodydim).to(device)
    for i in range(num_envs):
        for joint in [t[i]]:
            bodies[i, joint] = (
                random.random() * upper_bound if flawed_rate == -1 else flawed_rate
            )
    return bodies, t

def step_body(bodies, joint, rate = 0.004, threshold = 0, upper_bound=1): #each joint has a flaw rate to be partial of itself.
    '''
    joint: (num_envs, num) OR a single int, 每个环境对应的1个坏损关节 
        #TODO: joint will become (num_envs, num), num means the number of flawed joints.
    rate: 每个step, 有rate的概率使得关节扭矩往下掉, 剩余扭矩比例随机
    threshold, 在剩余扭矩低于threshold时, 重置到随机的一个扭矩。
    '''        
    num_envs = bodies.shape[0]
    t = torch.rand(num_envs)
    t = (t<rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodies.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodies.device)
        # print(bodies.shape, joint.shape, t.shape)
        p = torch.gather(bodies, 1, joint) * t
        bodies = torch.scatter(bodies, 1, joint, p)
        if threshold > 0: 
            # tmp = torch.gather(bodies, 1, joint)
            # t = (tmp < threshold) * torch.rand(num_envs, device=bodies.device)
            # t = t.to(bodies.device)
            # t = 1 / (1 - t)
            # bodies = torch.scatter(bodies, 1, joint, t * tmp)
            rands = torch.rand_like(bodies)/2 + 0.6
            rands = torch.clamp(rands, min=0, max=upper_bound-1e-9)
            bodies = torch.where(bodies>threshold, bodies, rands)
            # bodies = torch.clamp(bodies, min=0, max=upper_bound)
    else:
        bodies[:, joint] *= t
        if threshold > 0:  # Here we assume that joint must be a single int
            # t = (bodies[:, joint] < threshold) * torch.rand(num_envs, device=bodies.device) * (torch.rand(num_envs, device= bodies.device) < rate_reset)
            # t = t.to(bodies.device)
            # t = 1 / (1 - t)
            # bodies[:, joint] *= t
            rands = torch.rand_like(bodies)/2 + 0.6
            rands = torch.clamp(rands, min=0, max=upper_bound-1e-9)
            bodies = torch.where(bodies>threshold, bodies, rands)
            # bodies = torch.clamp(bodies, min=0, max=upper_bound)

    return bodies

def evaluate_on_env_batch_body(
    model,
    device,
    context_len,
    env,
    body_target,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
    prompt_policy=None,
):
    """
    用来测试EAT及其变种的模型表现
    ------------------
    model：待测试模型
    device：测试硬件号
    context_len：模型上下文长度
    env：测试环境
    body_target：目前除了用来提供bodydim外暂时无用
    num_eval_ep=10：测试所用环境数量
    max_test_ep_len=1000：每个环境中测试长度
    state_mean=None：如其名
    state_std=None：如其名
    body_mean=None：如其名，无用
    body_std=None：如其名，无用
    render=False：是否渲染，暂时无用
    prompt_policy：暂时无用
    body_pre=False：是否将自身预测的body作为下一帧的body输入
    body_gt：是否将真实body作为下一帧预测的输入（body_pre为true时为body预测的输入，否则为action预测的输入）
    """

    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    body_dim = len(body_target)

    assert state_mean is not None
    if not torch.is_tensor(state_mean):
        state_mean = torch.from_numpy(state_mean).to(device)
    else:
        state_mean = state_mean.to(device)

    assert state_std is not None
    if not torch.is_tensor(state_std):
        state_std = torch.from_numpy(state_std).to(device)
    else:
        state_std = state_std.to(device)

    model.eval()

    with torch.no_grad():
        actions = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        states = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )  # Here we assume that obs = state !!
        running_body, joints = flaw_generation(
            eval_batch_size, bodydim=body_dim, fixed_joint=[-1], device=device
        )
        running_body = running_body.to(device)
        bodies = running_body.expand(max_test_ep_len, eval_batch_size, body_dim).type(
            torch.float32
        )
        bodies = torch.transpose(bodies, 0, 1).to(device)
        running_state, _ = env.reset()

        total_rewards = torch.zeros(eval_batch_size).to(device)
        steps = torch.zeros(eval_batch_size).to(device)
        dones = torch.zeros(eval_batch_size).to(device)

        for t in range(max_test_ep_len):
            total_timesteps += 1
            states[:, t, :] = running_state
            states[:, t, :] = (states[:, t, :] - state_mean) / state_std
            bodies[:, t, :] = running_body

            if t < context_len:
                _, act_preds, _ = model.forward(
                    states[:, :context_len],
                    actions[:, :context_len],
                    bodies=bodies[:, :context_len],
                )

                if prompt_policy is None:
                    act = act_preds[:, t].detach()
                else:
                    act = prompt_policy(running_state.to(device))
            else:
                
                _, act_preds, _ = model.forward(
                    states[:, t - context_len + 1 : t + 1],
                    actions[:, t - context_len + 1 : t + 1],
                    bodies=bodies[:, t - context_len + 1 : t + 1],
                )
                act = act_preds[:, -1].detach()
                
            running_state, _, running_reward, done, infos, _ , _ = env.step(
                act, running_body
            )

            actions[:, t] = act
            total_rewards += (
                running_reward
                * (dones == 0)
            ).detach()
            steps += (dones==0).detach()

            dones += done.detach()

            running_body = step_body(running_body, joints, rate=0.004, threshold=0.0001, upper_bound=0.5)
    results["eval/avg_reward"] = torch.mean(total_rewards)
    results["eval/avg_ep_len"] = torch.mean(steps)

    return results

def parallel_load(path):
    start = time.time()
    print("START LOADING ......")

    def process(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    p = multiprocessing.Pool()
    res_list = []
    for f in glob.glob(os.path.join(path, "*.pkl")):
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        res = p.apply_async(process, [f])
        res_list.append(res)

    p.close()
    p.join()  # Wait for all child processes to close.

    # print("======================")
    data_list = []
    for res in res_list:
        print("PROCESSING  ", res)
        data_list.append(res.get())
    end = time.time()
    print(f"FINISHED LOADING!! ({end - start}) SEC")
    # pdb.set_trace()
    return data_list


def load_path(path):
    print("START LOADING ......")

    data_list = []

    for file in tqdm(glob.glob(os.path.join(path, "*.pkl"))):
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        data_list.append(dataset)

    # pdb.set_trace()
    return data_list


class D4RLTrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        context_len,
        leg_trans=False,
        leg_trans_pro=False,
        bc=False,
    ):

        self.context_len = context_len

        # load dataset
        if type(dataset_path) == str:
            if dataset_path.endswith(".pkl"):
                with open(dataset_path, "rb") as f:
                    self.trajectories = pickle.load(f)
            else:
                self.trajectories = load_path(dataset_path)

        elif type(dataset_path) == list:
            self.trajectories = dataset_path

        if bc:
            print("CONCATENATE BODY INTO STATE ..........")
            for traj in tqdm(self.trajectories):
                traj["observations"] = np.concatenate(
                    [traj["bodies"], traj["observations"]], axis=1
                )
                traj["next_observations"] = np.concatenate(
                    [traj["bodies"], traj["next_observations"]], axis=1
                )

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj["observations"].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj["observations"])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )
        states = []
        # normalize states
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

    def get_state_stats(self, body=False):
        return self.state_mean, self.state_std

    @property
    def body_dim(self):
        traj = self.trajectories[0]
        return traj.get("bodies", traj.get("bodys", traj.get("body", torch.ones(12)))).shape[-1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj["observations"][si : si + self.context_len])
            actions = torch.from_numpy(traj["actions"][si : si + self.context_len])
            bodies = torch.from_numpy(
                traj.get("bodies", traj.get("bodys", traj.get("body", torch.ones(traj_len, 12))))[si : si + self.context_len]
            )
            gt_bodies = torch.from_numpy(
                traj.get("gt_bodies", traj.get("gt_bodys", traj.get("bodies", traj.get("bodys", traj.get("body", torch.ones(traj_len, 12))))))[si : si + self.context_len]
            )

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])), dtype=states.dtype
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])), dtype=actions.dtype
                    ),
                ],
                dim=0,
            )

            bodies = torch.from_numpy(
                        traj.get("bodies", traj.get("bodys", traj.get("body", None)))
                        )            
            bodies = torch.cat(
                [
                    bodies,
                    torch.zeros(
                        ([padding_len] + list(bodies.shape[1:])),
                        dtype=bodies.dtype,
                    ),
                ],
                dim=0,
            )

            gt_bodies = torch.from_numpy(
                traj.get("gt_bodies", traj.get("gt_bodys", traj.get("bodies", traj.get("bodys", traj.get("body", None)))))
            )
            gt_bodies = torch.cat(
                [
                    gt_bodies,
                    torch.zeros(
                        ([padding_len] + list(gt_bodies.shape[1:])),
                        dtype=gt_bodies.dtype,
                    ),
                ],
                dim=0,
            )
            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return states, actions, bodies, gt_bodies, traj_mask

def get_dataset_config(dataset):

    eval_env = "none"
    datafile = ""
    i_magic_list = []
    eval_body_vec = []

    if dataset == "IPPOtest":
        datafile = "Trajectory_IPPO_3"
        i_magic_list = [f"PPO_I_{x}" for x in range(1)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO":
        datafile = "Trajectory_IPPO"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO2":
        datafile = "Trajectory_IPPO_2"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO3":
        datafile = "Trajectory_IPPO_3"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO6":
        datafile = "Trajectory_IPPO_6"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO8":
        datafile = "Trajectory_IPPO_8"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPO9":
        datafile = "Trajectory_IPPO_9"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPOUB3":
        datafile = "Trajectory_IPPO_UB3"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "IPPOUB3_2":
        datafile = "Trajectory_IPPO_UB3_2"
        i_magic_list = [f"PPO_I_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody":
        datafile = "Trajectory_Ebody"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody2":
        datafile = "Trajectory_Ebody2"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "EBody3":
        datafile = "Trajectory_Ebody3"
        i_magic_list = [f"EBody_v1_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "AMP_PPO":
        datafile = "Trajectory_AMP"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
        
    if dataset == "AMP_PPO_no_liny":
        datafile = "Trajectory_AMP_No_liny"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "AMP_PPO_latency30":
        datafile = "Trajectory_AMP_latency30"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "New_URDF":
        datafile = "New_URDF"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "New_URDF_Step":
        datafile = "New_URDF_Step"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "New_URDF_Step_2":
        datafile = "New_URDF_Step_More"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    
    if dataset == "Small_Damp_Step":
        datafile = "Small_Damp_Step"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]

    if dataset == "No_Damp_Step":
        datafile = "No_Damp_Step"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "No_Damp_Step_2":
        datafile = "No_Damp_Step_2"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Small_Damp_New_Reward":
        datafile = "Small_Damp_New_Reward"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Mix_Damp":
        datafile = "Mix_Damp"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(39)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Small_Damp_Noise":
        datafile = "Small_Damp_Noise"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Small_Damp_Noise_No_Push":
        datafile = "Small_Damp_Noise_No_Push"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(13)]
        eval_body_vec = [1 for _ in range(12)]
    if dataset == "Small_Damp_Noise_No_Push_2":
        datafile = "Small_Damp_Noise_No_Push_2"
        i_magic_list = [f"PPO_AMP_{x}" for x in range(12)]
        eval_body_vec = [1 for _ in range(12)]
    

    return datafile, i_magic_list, eval_body_vec, eval_env
