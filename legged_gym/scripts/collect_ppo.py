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
ZERO_VEL = False
VEL_TO_ACC = False
NOISE = True
TOPK = 0
THRESHOLD = 0.0001
PASSSCORE = 11.5 # 12 #24 #12 #24	#设定分数下界   
UPPERBOUND = 1
STEP_RATE = 0.02
MAX_EPISODE_LEN = 500 #1000
VEL0_5 = False
VEL0_4 = False
SAVE_DIR = os.path.join(parentdir, "data")
if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)
DATASET_NAME = "Small_Damp_Noise_No_Push_2"
# DATASET_NAME = "Trajectory_AMP_No_liny_Few_Step"
BIAS = 0
TEACHER = "Small_Damp_No_push"
# param_dict = {}
## 断腿版本的i_magic命名规则：共12维，1代表正常 小数代表力矩折扣率
## 12维依次是左前右前左后右后腿 从上到下的关节顺序
## 注释中代号：L为左 R为右 F为前 B为后 H为臀 K为膝 A为踝 出现字母代表对应位置坏损
codename_list = []	#存储每条腿的字母代号
for i in ["F", "B"]:
    for j in ["L", "R"]:
        for k in ["H", "K", "A"]:
            codename_list.append(j+i+k)
# rate_list = [0, 0.25, 0.5, 0.75]#储存现有的故障比率模型

def play(args, env, train_cfg, fault_id = -1, fault_rate_upperbound = 1, bias = 0):
	'''
	用以采集某种具体足部故障模式的数据，需要输入故障关节字母表示和故障率
	----------------------------
	输入参数:
	args: 预设参数集合
	env: 采集数据的环境
	train_cfg: 模型信息
	fault_type: 故障关节字母表示，none代表完好无损
	fault_rate: 故障关节比率，1代表完整力矩，小数代表力矩折扣
	----------------------------
	'''
	
	# load policy
	train_cfg.runner.resume = True
	train_cfg.runner.load_run = TEACHER
	train_cfg.runner.checkpoint = fault_id
	train_cfg.runner.experiment_name = "Models"
	if fault_id==-1:
		train_cfg.runner.load_run = TEACHER
		train_cfg.runner.checkpoint = 1111
        # train_cfg.runner.load_run = "model_" + str(fault_id)

		# train_cfg.runner.load_run = fault_type + "_" + str(fault_rate)
		# model_root = os.path.join(model_root, str(fault_rate) + "_torques")
		#判断模型文件是否存在 若不存在则报错弹出
		# if not os.path.exists(os.path.join(model_root,train_cfg.runner.load_run)):
		# 	print(f"no model file{fault_type}_{fault_rate}")
		# 	return fault_type + "_" + str(fault_rate) + " file not exists"
	ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, evaluate=True)
	policy = ppo_runner.get_inference_policy(device=env.device)


	output_file = os.path.join(SAVE_DIR, DATASET_NAME)
	if not os.path.exists(output_file):
		os.mkdir(output_file)
	file_name = f"PPO_AMP_{fault_id+bias}.pkl"
	if fault_id==-1:
		file_name = f"PPO_AMP_{12+bias}.pkl"
	# special = ""
	# if ZERO_VEL:
	# 	special += "-zerovel"
	# if VEL0_5:
	# 	special += "-vel0.5"
	# if VEL0_4:
	# 	special += "-vel0.4"
	# if NOISE:
	# 	special += "-n"
    		
	# OUTPUT_FILE = "DTdata/a1magic4-uncle-v2.pkl"
	# ver_index = 0
	# if not TOPK:
	# 	output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# else:
	# 	output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# if not PASSSCORE:
	# 	output_file = f"{SAVE_DIR}{fault_type}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# else:
	# 	output_file = f"{SAVE_DIR}{fault_type}-P{PASSSCORE}F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"

	# while os.path.isfile(output_file):
	# 	ver_index += 1
	# 	if not TOPK:
	# 		output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-F{NUM_ENVS*REP}{special}-v{ver_index}.pkl"
	# 	else:
	# 		output_file = f"{SAVE_DIR}{fault_type}-{fault_rate}-{TOPK}outof{NUM_ENVS*REP}{special}-v{ver_index}.pkl"

	print("Preparing file to ", os.path.join(output_file , file_name))

	total_rewards = np.zeros(NUM_ENVS)
	total_dones = np.zeros(NUM_ENVS)

	print("RECORDING DATA ......")
	# assert int(env.max_episode_length) == 1000
	env.max_episode_length = MAX_EPISODE_LEN
	print(env.max_episode_length)
	paths = []
	body_latency = args.body_latency
	for _ in trange(REP):
		data_set = {'observations':[], 'bodies':[],  'gt_bodies':[], 'actions':[], 'terminals':[], 'rewards':[]}

		total_rewards = np.zeros(NUM_ENVS)
		total_dones = np.zeros(NUM_ENVS)
		bodies, joints = flaw_generation(env.num_envs, bodydim=12, fixed_joint=[fault_id], device=env.device, upper_bound=fault_rate_upperbound)
		body_buffer = [bodies] * (body_latency+1)
		length_body_buffer = body_latency + 1
		cur_body_index = 0
		env.reset()
		obs = env.get_observations()
		total_dones = np.zeros(NUM_ENVS)
		for i in trange(int(env.max_episode_length)):
			actions = policy(obs.detach(), body_buffer[cur_body_index])
			obs_ori = obs.cpu().detach().numpy()[:,:48]
			# GIVE THE BODY WITH LATENCY, NOT THE REAL-TIME BODY
			bodies_ori = body_buffer[cur_body_index].cpu().detach().numpy() 
			obs, _ , rews, dones, infos, _, _ = env.step(actions.detach(), bodies)
		

			

			data_set['observations'].append(obs_ori)
			data_set['bodies'].append(bodies_ori)
			data_set['gt_bodies'].append(bodies.cpu().detach().numpy())
			data_set['rewards'].append(rews.cpu().detach().numpy())
			data_set['terminals'].append(dones.cpu().detach().numpy())
			# data_set['next_observations'].append(obs.cpu().detach().numpy()[:,:48])
			# data_set['timeouts'].append(infos["time_outs"].cpu().detach().numpy())
			# data_set['timeouts'][-1] = np.array([True]*NUM_ENVS) if i == int(env.max_episode_length-1) else data_set['timeouts'][-1]
			data_set['actions'].append(actions.cpu().detach().numpy())
			total_rewards += rews.detach().cpu().numpy() * (total_dones == 0)
			total_dones += dones.detach().cpu().numpy()
			
			if fault_id == -1:
				bodies = step_body(bodies, joints, rate = STEP_RATE, threshold= 0.15, upper_bound=fault_rate_upperbound)
			else:
				bodies = step_body(bodies, joints, rate = STEP_RATE, threshold= THRESHOLD, upper_bound=fault_rate_upperbound)


			body_buffer[cur_body_index] = bodies
			cur_body_index += 1
			if cur_body_index == length_body_buffer:
				cur_body_index = 0
		print("MEAN SCORE: ", np.mean(total_rewards))

		print("[REORGANISING DATA ......]")

		# keys = ["observations", "actions", "rewards", "terminals"]

		# for k in keys:
		# 	print("Preprocessing ", k)
		# 	data_set[k] = np.array(data_set[k])


		obss = np.array(data_set['observations']).transpose((1,0,2))
		# nobss = np.array(data_set['next_observations']).transpose((1,0,2))
		bodies = np.array(data_set['bodies']).transpose((1,0,2))
		gt_bodies = np.array(data_set['gt_bodies']).transpose((1,0,2))
		acts = np.array(data_set['actions']).transpose((1,0,2))
		ds = np.array(data_set['terminals']).transpose()
		rs = np.array(data_set['rewards']).transpose()
		print(obss.shape)
		# input()
		for obs_p, bodies_p, gt_bodies_p, act_p, rew_p, done_p in zip(obss, bodies, gt_bodies, acts, rs, ds):
			obs_list = []
			bodies_list = []
			gt_bodies_list = []
			# nobs_list = []
			act_list = []
			rew_list = []
			done_list = []
			path_dict = {}

			for obs_t, bodies_t, gt_bodies_t, act_t, rew_t, done_t in zip(obs_p, bodies_p, gt_bodies_p, act_p, rew_p, done_p):
				obs_list.append(obs_t)
				bodies_list.append(bodies_t)
				gt_bodies_list.append(gt_bodies_t)
				# nobs_list.append(nobs_t)
				act_list.append(act_t)
				rew_list.append(rew_t)
				done_list.append(done_t)
				if done_t:
					break

			path_dict['observations'] = np.array(obs_list)
			path_dict['bodies'] = np.array(bodies_list)
			path_dict['gt_bodies'] = np.array(gt_bodies_list)
			# path_dict['next_observations'] = np.array(nobs_list)
			path_dict['actions'] = np.array(act_list)
			path_dict['rewards'] = np.array(rew_list)
			path_dict['terminals'] = np.array(done_list)

		# embodiment = [1 for _ in range(12)]
		# if fault_type != "none":
		# 	embodiment[codename_list.index(fault_type)] = fault_rate #将坏损关节的embody置为坏损率
		# path_dict["body"] = np.tile(np.array(embodiment), (np.shape(path_dict["observations"])[0], 1))

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
		     "gt_bodies": paths[i]['gt_bodies'], \
			 "observations": paths[i]['observations'] } 
		)

	print("-->")

	# returns = np.array([np.sum(p['rewards']) for p in paths_out])
	num_samples = np.sum([p['actions'].shape[0] for p in paths_out])
	print(f'Number of samples collected: {num_samples}')

	if num_samples == 0:
		print("NO USEFUL TRAJECTORIES !")
		return str(fault_id) + "_"  + "no traj"
	with open(os.path.join(output_file, "log.out"), 'a') as log:
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
	env_cfg.env.num_envs = NUM_ENVS # min(env_cfg.env.num_envs, NUM_ENVS)
	env_cfg.terrain.num_rows = 5
	env_cfg.terrain.num_cols = 5
	env_cfg.terrain.curriculum = False
	env_cfg.noise.add_noise = NOISE # False
	env_cfg.domain_rand.randomize_friction = NOISE # False
	# env_cfg.domain_rand.push_robots = NOISE # False
	env_cfg.domain_rand.push_robots = False

	# Fast
	env_cfg.commands.ranges.lin_vel_x = [0, 1]
	env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
	env_cfg.commands.ranges.ang_vel_yaw = [-0.5,0.5]

	# # No lin_y
	# env_cfg.commands.ranges.lin_vel_x = [-0.7,0.7]
	# env_cfg.commands.ranges.lin_vel_y = [-0,0]
	# env_cfg.commands.ranges.ang_vel_yaw = [-1,1]

	# if ZERO_VEL:
	# 	env_cfg.commands.ranges.lin_vel_x = [0.2,0.5]
	# 	env_cfg.commands.ranges.lin_vel_y = [0.0,0.15]
	# 	env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.15]

	# if VEL0_5:
	# 	env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
	# 	env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
	# 	env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

	# if VEL0_4:
	# 	env_cfg.commands.ranges.lin_vel_x = [0.4,0.4]
	# 	env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
	# 	env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]

	env_cfg.terrain.mesh_type = 'plane'
	output_file = os.path.join(SAVE_DIR, DATASET_NAME)
	if (not os.path.exists(output_file)):
		os.mkdir(output_file)
	with open(os.path.join(output_file, "log.out"), 'a') as f:
		print("TEACHER:", TEACHER, file = f)
		print("Episode_len: ", MAX_EPISODE_LEN, file = f)
		print("REP: ", REP, file = f)
		print("UpperBound: ", UPPERBOUND, file = f)
		print("Step Rate:", STEP_RATE, file = f)
		print("Threshold: ", THRESHOLD, file = f)
		print("PASSSCORE:", PASSSCORE, file = f)
		print("Num_ENVS:", NUM_ENVS, file = f)
		print("ENV_CFG:", env_cfg)
	# prepare environment
	env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
	failed_set = set()
	#采集所有坏关节情况的数据
	# for name in codename_list:
	# 	for rate in rate_list:
	# 		failed_set.add(play(args, env, train_cfg, name, rate))
	for joint in range(-1, 12):
	# for joint in range(2,3):
		failed_set.add(play(args, env, train_cfg, fault_id=joint, fault_rate_upperbound=UPPERBOUND, bias=BIAS))
	# play(args, env, train_cfg)	#采集四条腿都能用的机器狗的数据
	# play(args, env, train_cfg, "RBK", 0)
	print(failed_set)