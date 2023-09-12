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

import pdb
import pickle
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
# os.sys.path.insert(0, os.path.dirname(currentdir))

import isaacgym
from legged_gym.envs import *

from legged_gym import LEGGED_GYM_ROOT_DIR
import collections
from legged_gym.utils import  get_args, export_policy_as_jit, Logger
from legged_gym.utils import task_registry
import numpy as np
import torch

from tqdm import trange, tqdm
from legged_gym.scripts.utils import flaw_generation, step_body


NUM_ENVS = 10000 #10000 #5000 #10000 #4000 #10000 # 400 #4000 #1000 # 50# 20000 #
REP = 2# 1 5 #10 #20
NOISE = True
TOPK = 0
THRESHOLD = 0.0001
PASSSCORE = 11.5 # 12 #24 #12 #24	#setting the lowerbound of passing return of a trajectory.
UPPERBOUND = 1
STEP_RATE = 0.02
MAX_EPISODE_LEN = 500 #1000

SAVE_DIR = os.path.join(parentdir, "data")
if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)

def play(args, env, train_cfg, fault_id = -1, fault_rate_upperbound = 1):
	'''
	fault_id: id of the joint whose actuator suffering degradation
	fault_rate_upperbound: upperbound of the degradation rate
	'''	
	# load policy
	train_cfg.runner.resume = True
	train_cfg.runner.load_run = args.load_run
	train_cfg.runner.checkpoint = args.checkpoint
	train_cfg.runner.experiment_name = args.experiment_name
	ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, evaluate=True)
	policy = ppo_runner.get_inference_policy(device=env.device)

	output_file = os.path.join(SAVE_DIR, args.file_name)
	if not os.path.exists(output_file):
		os.mkdir(output_file)
	file_name = f"Joint_{fault_id}.pkl"

	print("Preparing file to ", os.path.join(output_file , file_name))

	total_rewards = np.zeros(NUM_ENVS)
	total_dones = np.zeros(NUM_ENVS)

	print("RECORDING DATA ......")
	env.max_episode_length = MAX_EPISODE_LEN
	paths = []
	for _ in trange(REP):
		data_set = {'observations':[], 'bodies':[], 'actions':[], 'terminals':[], 'rewards':[]}

		total_rewards = np.zeros(NUM_ENVS)
		total_dones = np.zeros(NUM_ENVS)
		bodies, joints = flaw_generation(env.num_envs, bodydim=12, fixed_joint=[fault_id], device=env.device, upper_bound=fault_rate_upperbound)
		env.reset()
		obs = env.get_observations()
		total_dones = np.zeros(NUM_ENVS)
		for i in trange(int(env.max_episode_length)):
			actions = policy(obs.detach(), bodies)
			obs_ori = obs.cpu().detach().numpy()[:,:48]

			bodies_ori = bodies.cpu().detach().numpy() 
			obs, _ , rews, dones, infos, _, _ = env.step(actions.detach(), bodies)
		

			data_set['observations'].append(obs_ori)
			data_set['bodies'].append(bodies_ori)
			data_set['rewards'].append(rews.cpu().detach().numpy())
			data_set['terminals'].append(dones.cpu().detach().numpy())
			data_set['actions'].append(actions.cpu().detach().numpy())
			total_rewards += rews.detach().cpu().numpy() * (total_dones == 0)
			total_dones += dones.detach().cpu().numpy()
			
			if fault_id == -1:
				bodies = step_body(bodies, joints, rate = STEP_RATE, threshold= 0.15, upper_bound=fault_rate_upperbound)
			else:
				bodies = step_body(bodies, joints, rate = STEP_RATE, threshold= THRESHOLD, upper_bound=fault_rate_upperbound)

		print("MEAN SCORE: ", np.mean(total_rewards))

		print("[REORGANISING DATA ......]")


		obss = np.array(data_set['observations']).transpose((1,0,2))
		bodies = np.array(data_set['bodies']).transpose((1,0,2))
		acts = np.array(data_set['actions']).transpose((1,0,2))
		ds = np.array(data_set['terminals']).transpose()
		rs = np.array(data_set['rewards']).transpose()
		for obs_p, bodies_p, act_p, rew_p, done_p in zip(obss, bodies, acts, rs, ds):
			obs_list = []
			bodies_list = []
			act_list = []
			rew_list = []
			done_list = []
			path_dict = {}

			for obs_t, bodies_t, act_t, rew_t, done_t in zip(obs_p, bodies_p, act_p, rew_p, done_p):
				obs_list.append(obs_t)
				bodies_list.append(bodies_t)
				act_list.append(act_t)
				rew_list.append(rew_t)
				done_list.append(done_t)
				if done_t:
					break

			path_dict['observations'] = np.array(obs_list)
			path_dict['bodies'] = np.array(bodies_list)
			path_dict['actions'] = np.array(act_list)
			path_dict['rewards'] = np.array(rew_list)
			path_dict['terminals'] = np.array(done_list)

			paths.append(path_dict)


	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['actions'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	# # pdb.set_trace()
	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return str(fault_id) + "_" + "no traj"

	if not PASSSCORE:
		top10000 = (-returns).argsort()[:TOPK]
	else:
		top10000 = np.nonzero(returns > PASSSCORE)[0]
	paths_out = []
	for i in tqdm(top10000):
		paths_out.append(
			{"actions": paths[i]['actions'], \
    		 "bodies": paths[i]['bodies'], \
			 "observations": paths[i]['observations'] } 
		)

	print("-->")

	# returns = np.array([np.sum(p['rewards']) for p in paths_out])
	num_samples = np.sum([p['actions'].shape[0] for p in paths_out])
	print(f'Number of samples collected: {num_samples}')

	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return str(fault_id) + "_"  + "no traj"
	with open(os.path.join(output_file, "log.txt"), 'a') as log:
		print(f'Trajectory {file_name}: returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}', file = log)
		print(f'Number of samples collected: {num_samples}', file=log)

	with open(os.path.join(output_file, file_name), 'wb') as f:
		pickle.dump(paths_out, f)
	print("Saved to ", os.path.join(output_file, file_name), " ~!")
	return ""



if __name__ == '__main__':
	
	args = get_args()
	env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
	# override some parameters for testing
	env_cfg.env.num_envs = NUM_ENVS
	env_cfg.terrain.num_rows = 5
	env_cfg.terrain.num_cols = 5
	env_cfg.terrain.curriculum = False
	env_cfg.noise.add_noise = NOISE
	env_cfg.domain_rand.randomize_friction = NOISE # False

	env_cfg.commands.ranges.lin_vel_x = [0, 1]
	env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
	env_cfg.commands.ranges.ang_vel_yaw = [-0.5,0.5]

	env_cfg.terrain.mesh_type = 'plane'
	output_file = os.path.join(SAVE_DIR, args.file_name)
	if (not os.path.exists(output_file)):
		os.mkdir(output_file)
	with open(os.path.join(output_file, "log.txt"), 'a') as f:
		print("Episode_len: ", MAX_EPISODE_LEN, file = f)
		print("REP: ", REP, file = f)
		print("UpperBound: ", UPPERBOUND, file = f)
		print("Step Rate:", STEP_RATE, file = f)
		print("Threshold: ", THRESHOLD, file = f)
		print("PASSSCORE:", PASSSCORE, file = f)
		print("Num_ENVS:", NUM_ENVS, file = f)
	# prepare environment
	env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

	play(args, env, train_cfg, fault_id=args.joint, fault_rate_upperbound=UPPERBOUND)