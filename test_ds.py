import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_agents.agent import Agent

from torch.utils.data import DataLoader, Subset
from ope_methods.fqe import QFitterBase, QFitterL2, QFitterLME, QFitterDD
from ope_methods.dataset import F110Dataset, F110DatasetSequence, random_split_indices, model_split_indices
from ope_methods.iw import ImportanceSamplingContinousStart
import ope_methods
from functools import partial
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse    


F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        set_terminals=False,
        delta_factor=1.0,
        reward_config="reward_progress.json",
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = True,
        include_progress=False,
        set_previous_step_terminals=0,
        use_compute_termination=True,
        remove_cons_terminals=True,
        **dict(name="f110-real-stoch-v2",
            config = dict(map="Infsaal3", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )

dataset = F110Env.get_dataset(train_only=True)
print(np.unique(dataset["model_name"]))
eval_ds = F110Env.get_dataset(eval_only=True)
print(np.unique(eval_ds["model_name"]))
exit()
trajectories, actions, terminations , model_names = F110Env.compute_trajectories(dataset["observations"], 
                                                                                 dataset["actions"], 
                                                                                 dataset["terminals"],
                                                                                 dataset["timeouts"], 
                                                                                 dataset["model_name"])
print(terminations)
F110Env.plot_trajectories(trajectories#readd dimension
                              , model_names, terminations)
should_throw = np.where(terminations < 10)[0]
print(should_throw)
print(trajectories[should_throw])
print(model_names[should_throw])
for model_name in np.unique(model_names):
    only_model = np.where(model_names == model_name)[0]
    F110Env.plot_trajectories(trajectories[only_model] #readd dimension
                              , model_names[only_model], terminations[only_model])