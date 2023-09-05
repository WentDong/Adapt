"""
本脚本用于测试从ippo采集到的数据，joint的坏损分布如何
"""

import os
import pickle
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

import yaml
from tqdm import tqdm
from utils import get_dataset_config

import matplotlib.pyplot as plt
import numpy as np


def count_traj(args):

    args["dataset"] = "New_URDF"

    datafile, i_magic_list, _, _ = get_dataset_config(args["dataset"])

    # file_list = [f"a1magic{i_magic}-{datafile}.pkl" for i_magic in i_magic_list]
    file_list = [os.path.join(datafile, f"{i_magic}.pkl") for i_magic in i_magic_list]
    dataset_path_list_raw = [os.path.join(args["dataset_dir"], d) for d in file_list]
    dataset_path_list = []
    for p in dataset_path_list_raw:
        if os.path.isfile(p):
            dataset_path_list.append(p)
        else:
            print(p, " doesn't exist~")

    # 加载轨迹部分
    body_vec = []
    for pkl in tqdm(dataset_path_list):
        with open(pkl, "rb") as f:
            thelist = pickle.load(f)

        bodies = []
        for traj in thelist:
            bodies.append(traj["bodies"])
        body_vec.append(
            np.concatenate(bodies, axis=0).transpose()[
                int(pkl.split(".")[0].split("_")[-1])
            ]
        )  # 仅保留有折损的body列
    # bodies = np.concatenate(bodies, axis=0).transpose()

    # draw
    print("drawing . . .")
    _, ax = plt.subplots()

    vp = ax.violinplot(
        dataset=body_vec,
        positions=range(len(body_vec)),  # widths=0.5,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    # styling:
    for body in vp["bodies"]:
        body.set_alpha(0.9)
    ax.set(
        xlim=(-0.5, 11.5),
        xticks=np.arange(0, 12),
        ylim=(-0.05, 1.05),
        yticks=np.arange(0, 1.05, 0.05),
    )

    plt.savefig(
        os.path.join("./Integration_EAT/scripts/tools/traj_analyse/", args["dataset"]),
        dpi=150,
    )
    print("fig saved")
    return


if __name__ == "__main__":
    with open("./Integration_EAT/scripts/args.yaml", "r") as fargs:
        args = yaml.safe_load(fargs)
    count_traj(args)
