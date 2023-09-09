import glob
import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def flaw_generation(
    num_envs, bodydim=12, fixed_joint=[-1], flawed_rate=-1, device="cpu", upper_bound=1
):
    """
    num_envs: parallel envs
    fixed_joint: id of joint, -1 for randomization
    flawed_rate: degeneration rate, -1 for randomization

    Outputs: bodies, joints
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
    joint: (num_envs, num) OR a single int
    rate: In every step, w.p. rate to degenerate the joint worse.
    threshold: if the degenerate rate is lower than threshold, it will be set to a random value between 0.5 and 1.
    '''       
    num_envs = bodies.shape[0]
    t = torch.rand(num_envs)
    t = (t<rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodies.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodies.device)
        p = torch.gather(bodies, 1, joint) * t
        bodies = torch.scatter(bodies, 1, joint, p)
        if threshold > 0: 
            rands = torch.rand_like(bodies)/2 + 0.6
            rands = torch.clamp(rands, min=0, max=upper_bound-1e-9)
            bodies = torch.where(bodies>threshold, bodies, rands)
    else:
        bodies[:, joint] *= t
        if threshold > 0:  
            rands = torch.rand_like(bodies)/2 + 0.6
            rands = torch.clamp(rands, min=0, max=upper_bound-1e-9)
            bodies = torch.where(bodies>threshold, bodies, rands)

    return bodies

def evaluate_on_env_batch_body(
    model,
    device,
    context_len,
    env,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
):
    eval_batch_size = num_eval_ep  # required for forward pass

    results = {}
    total_timesteps = 0
    state_dim = env.cfg.env.num_observations
    act_dim = env.cfg.env.num_actions
    body_dim = env.cfg.env.body_dim
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
                act = act_preds[:, t].detach()
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

    datafile = ""
    file_names = []
    if dataset == "Datas":
        datafile = "Datas"
        file_names = [f"Joint_{x}" for x in range(1)]
    

    return datafile, file_names
