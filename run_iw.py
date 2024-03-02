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
from create_plots import plot_bars_from_dicts
import json
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

parser = argparse.ArgumentParser(description='Run IW')


parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--save', action='store_true', help="save model")
parser.add_argument('--plot', action='store_true', help="plot results")
parser.add_argument('--log_prob_type', type=str, default="action", help="log prob type")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--type', type=str, default="WeightedIS", help="type of importance sampling")
args = parser.parse_args()


def compute_prob_trajectories(log_probs, finished, model_names, rewards):
    starts = np.where(np.roll(finished, 1) == 1)[0]
    ends = np.where(finished == 1)[0]
    horizon = np.max(ends + 1 -starts)
    prob_trajectories = np.zeros((len(starts), horizon, log_probs.shape[1]))
    terminations_ = np.zeros(len(starts))
    model_names_ = []
    rewards_ = np.zeros((len(starts), horizon))
    for i, (start, end) in enumerate(zip(starts, ends)):
        #print(start,end)
        #print(val_dataset.states[start:end+1].shape)
        prob_trajectories[i, 0:end - start+ 1] = log_probs[start:end+1]
        term = np.where(finished[start:end+1])[0]
        model_names_.append(model_names[start])
        if len(term )== 0:
            term = [horizon+1]
        rewards_[i, 0:end - start+ 1] = rewards[start:end+1]
        terminations_[i] = int(term[0])
    return prob_trajectories, terminations_, model_names_ ,rewards_


def main(args):



    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = f"runs_iw",
        algo=f"iw_{args.log_prob_type}",
        reward_name="reward_progress",
        dataset="f110-real-stoch-v2",
        target_policy="off-policy",
        seed = args.seed,
    )
    print(save_path)
    # append the args.agent to the save path
    # save_path = os.path.join(save_path, args.agent) 
    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")


    
    F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=args.target_reward,#"reward_progress.json",
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

    ### get the dataset ###
    training_dataset = F110Dataset(
        F110Env,
        normalize_states=False, # no normalization needed, we just need states to compute first IW step
        normalize_rewards=False,
        #eval_only = True, # for testing, todo! change
        #only_agents=["StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5"]
        train_only=True,
        #only_agents = [args.agent],
    )
    
    print("inital states training:", torch.sum(training_dataset.mask_inital))

    #print(target_log_probs)
    #print(training_dataset.log_probs)
    

    #print(training_dataset.next_actions[:20])
    #print(training_dataset.actions[:20])
    #exit()
    ### done ###
        #print(args)
    result_dict = {}
    #for agent_num, agent in enumerate(["StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5"]):#enumerate(F110Env.eval_agents):
    for agent_num, agent in enumerate(F110Env.eval_agents): 
        eval_dataset = F110Dataset(
            F110Env,
            normalize_states=True,
            normalize_rewards=False,
            only_agents = [agent],
            state_mean=training_dataset.state_mean,
            state_std=training_dataset.state_std,
            reward_mean=training_dataset.reward_mean,
            reward_std=training_dataset.reward_std,
        )

        # print(args.agent)
        if args.log_prob_type == "action":
            actor = Agent().load(name=agent, no_print=True)
            get_log_probs = partial(F110Env.get_target_log_probs, 
                                    actor=actor, 
                                    fn_unnormalize_states=training_dataset.unnormalize_states)
            # compute the next actions
            target_log_probs = get_log_probs(training_dataset.states,
                                            training_dataset.actions,
                                            scans = training_dataset.scans)

            behavior_log_probs = training_dataset.log_probs.reshape(training_dataset.log_probs.shape[0], -1)
        
        if args.log_prob_type == "state":
            from ope_methods.model_based import F110ModelBased, build_dynamics_model
            min_states = training_dataset.states.min(axis=0)[0]
            max_states = training_dataset.states.max(axis=0)[0]
            min_states[:2] = min_states[:2] - 2.0
            max_states[:2] = max_states[:2] + 2.0
            dynamics_model = build_dynamics_model("ProbsDeltaDynamicsModel",
                                                    min_state=min_states, 
                                                    max_state=max_states)
            model = F110ModelBased(F110Env, training_dataset.states.shape[1],
                            training_dataset.actions.shape[1],
                            dynamics_model = dynamics_model,
                            hidden_size = [256,256,256,256],
                            dt=1/20,
                            min_state=min_states,
                            max_state=max_states,
                            fn_normalize=training_dataset.normalize_states,
                            fn_unnormalize=training_dataset.unnormalize_states,
                            use_reward_model=False,
                            use_done_model=False,
                            obs_keys=training_dataset.obs_keys,
                            learning_rate=1e-3,
                            weight_decay=1e-4,
                            target_reward="reward_progress.json",
                            logger=None,)
            #if args.model_checkpoint is not None:
            model.load(f"/home/fabian/msc/f110_dope/ws_release/experiments/runs/ProbsDeltaDynamicsModel/f110-real-stoch-v2/250/off-policy/{args.seed}", 
                            f"model_{args.seed}_10000.pth")



            actor = Agent().load(name=agent)#"StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5")
            get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                        fn_unnormalize_states=training_dataset.unnormalize_states)
            target_actions = get_target_actions(training_dataset.states, 
                                                scans = training_dataset.scans, 
                                                deterministic=True)
            
            mean_target, logvar_target = model(training_dataset.states, target_actions, logvar=True)
            mean_behavior, logvar_behavior = model(training_dataset.states, training_dataset.actions, logvar=True)
            mean_target = mean_target.detach().numpy()
            mean_behavior = mean_behavior.detach().numpy()
            logvar_target = logvar_target.detach().numpy()
            logvar_behavior = logvar_behavior.detach().numpy()
            dims = mean_target.shape[1]
            std_dev_target = np.sqrt(np.exp(logvar_target))
            std_dev_behavior = np.sqrt(np.exp(logvar_behavior))
            #print(mean_target.shape)
            #print(std_dev_target.shape)
            #print(training_dataset.states_next.shape)
            # only get the relevant states_next
            from ope_methods.model_based import output_keys
            next_states = F110Env.get_specific_obs(training_dataset.states_next, output_keys)
            target_log_probs = norm.logpdf(next_states, mean_target, std_dev_target)
            behavior_log_probs = norm.logpdf(next_states, mean_behavior, std_dev_behavior)
            """
            for j in range(20):
                fig, axs = plt.subplots(1, dims, figsize=(20, 3))
                for i in range(target_log_probs.shape[1]):
                    x_range = np.linspace(min(mean_target[j, i], mean_behavior[j, i]) - 3*max(std_dev_target[j, i], std_dev_behavior[j, i]), 
                                            max(mean_target[j, i], mean_behavior[j, i]) + 3*max(std_dev_target[j, i], std_dev_behavior[j, i]), 100)
                    axs[i].plot(x_range, norm.pdf(x_range, mean_target[j, i], std_dev_target[j, i]), label='Target')
                    axs[i].plot(x_range, norm.pdf(x_range, mean_behavior[j, i], std_dev_behavior[j, i]), label='Behavior')
                    axs[i].set_title(f'Dimension {i+1}')
                    axs[i].legend()
                    axs[i].axvline(next_states[j,i], color='r', linestyle='--')
                
                print(target_log_probs[j], sum(target_log_probs[j]))
                print(behavior_log_probs[j], sum(behavior_log_probs[j]))
                print(sum(target_log_probs[j])-sum(behavior_log_probs[j]))
                plt.show()
            """
        #print(target_log_probs.shape)
        #print(behavior_log_probs.shape)

        start_points = training_dataset.states[training_dataset.mask_inital]
        # need to transform the log probs and target probs to trajectories
        finished = training_dataset.finished
        start_points_eval = eval_dataset.states[eval_dataset.mask_inital]
        #print(np.sum(finished))
        # terminations = training_dataset.terminations.numpy()
        behavior_log_probs, terminations_behavior, behavior_agent_names, rewards = compute_prob_trajectories(behavior_log_probs, finished, training_dataset.model_names, training_dataset.rewards)
        target_log_probs, terminatons_target, _ , rewards = compute_prob_trajectories(target_log_probs, finished, ["target"]*len(target_log_probs), training_dataset.rewards)
        # for the termination itself the value is still valid (not zero)
        #print(terminatons_target)

        #for i in range(behavior_log_probs.shape[0]):
        #print(behavior_log_probs.shape)
        # clip behavior and target log_probs 
        behavior_log_probs = np.clip(behavior_log_probs, -7, 2)
        target_log_probs = np.clip(target_log_probs, -7, 2)
        #plt.plot(behavior_log_probs[0].sum(axis=1))
        #plt.plot(target_log_probs[0].sum(axis=1))
            # plt.plot(target_log_probs[4])
        #plt.show()
        #print(terminations_behavior)
        # get minimum number of steps
        
        # TODO! add rescaling of log_probs? -> no longer principled? 
        # sum the behavior log probs along axis 2
        behavior_log_probs = np.sum(behavior_log_probs, axis=2)
        target_log_probs = np.sum(target_log_probs, axis=2)

        trajectories, actions, terminations , model_names = F110Env.compute_trajectories(training_dataset.states, training_dataset.actions, training_dataset.finished,training_dataset.finished, training_dataset.model_names)
        min_idx = np.argmin(terminations)

        #print(min_idx)
        #print(terminations[min_idx])
        #print(model_names[min_idx])
        #exit()
        # F110Env.plot_trajectories(trajectories, model_names, terminations)
        reward = ImportanceSamplingContinousStart(behavior_log_probs, 
                                        target_log_probs, 
                                        np.array([str(ag) for ag in behavior_agent_names]),
                                        terminations_behavior,
                                        rewards,
                                        start_points.numpy(),
                                        start_points_eval.numpy(),
                                        start_distance=1.0, 
                                        start_prob_method = "l2",
                                        plot=True,
                                        agent_name = agent)# args.plot,)
        print(f"Predicted {agent}: {reward}")
        result_dict[agent] = {"mean": reward, "std": 0.0}
    if args.plot:
        plot_bars_from_dicts([result_dict], ["Rollouts"], f"Mean discounted reward {args.target_reward}",plot=True)
    # add result to the save path
    save_path = os.path.join(save_path, "results")
    if args.save:
        for target_reward in args.target_reward:
         
            if args.save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path, f"{args.target_reward}"), 'w') as file:
                    json.dump(result_dict, file)
                

# main
if __name__ == "__main__":
    main(args)