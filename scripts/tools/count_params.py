
import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
from model import LeggedTransformerBody
import yaml
import numpy as np
import torch

model_name = "EAT_Small_Damp_SMALLDAMPNOISENOPUSH2_01"
run_name = "EAT_runs_AMP"
model_path = os.path.join(os.path.dirname(parentdir), run_name, model_name)
task_args = {}
with open(os.path.join(model_path, "args.yaml"), "r") as f:
	task_args_Loader = yaml.load_all(f, Loader = yaml.FullLoader)
	for t in task_args_Loader:
		task_args.update(t)
state_mean, state_std= np.load(os.path.join(model_path, "model.state_mean.npy")), np.load(os.path.join(model_path, "model.state_std.npy"))
body_mean, body_std = None, None

state_dim = 48
act_dim = 12
body_dim = 12

context_len = task_args["context_len"]      # K in decision transformer
n_blocks = task_args['n_blocks']            # num of transformer blocks
embed_dim = task_args['embed_dim']          # embedding (hidden) dim of transformer 
n_heads = task_args['n_heads']              # num of transformer heads
dropout_p = task_args['dropout_p']          # dropout probability
position_encoding_length = task_args['position_encoding_length']
device = torch.device("cpu")
pred_body = task_args.get('pred_body', True)
EAT_model = LeggedTransformerBody(
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
Num = sum(p.numel() for p in EAT_model.parameters())
print(Num)