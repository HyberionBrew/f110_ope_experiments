import f110_gym
import f110_orl_dataset
import gymnasium as gym

from f110_agents.agent import Agent

from torch.utils.data import DataLoader, Subset
from ope_methods.fqe import QFitterBase, QFitterL2, QFitterLME, QFitterDD
from ope_methods.dataset import F110Dataset, F110DatasetSequence, random_split_indices, model_split_indices

import ope_methods
from functools import partial
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import json


def get_infinite_iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

parser = argparse.ArgumentParser(description='Train model based approaches')

parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--eval_interval', type=int, default=5_000, help="eval interval")
parser.add_argument('--update_steps', type=int, default=200_000, help='update steps')
parser.add_argument('--split', type=str, default="on-policy", help="split")
parser.add_argument('--fqe_model', type=str, default="QFitterL2", help="dynamics model")
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--train', action='store_true', help="train")
parser.add_argument('--save_model', action='store_true', help="save model")
parser.add_argument('--skip_eval', action='store_true', help="skip eval")
parser.add_argument('--model_checkpoint', type=str, default=None, help="model checkpoint")
parser.add_argument('--agent',type=str, default="StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5", help="agent")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--tau', type=float, default=0.005, help="tau")
parser.add_argument('--weight_decay', type=str, default="Weight decay", help="dynamics model")
parser.add_argument('--eval_only', action='store_true', help="eval only")
args = parser.parse_args()


def sanity_check(F110Env, eval_dataset):
    print(eval_dataset.masks.sum())
    model_names = eval_dataset.model_names
    rewards = dict()
    min_reward, max_reward = 0, 0
    for model in np.unique(eval_dataset.model_names):
        model_trajectories = eval_dataset.states[model_names == model]
        model_action_trajectories = eval_dataset.actions[model_names == model]
        ends = np.where(eval_dataset.finished[model_names == model])[0]
        starts = np.where(eval_dataset.mask_inital[model_names == model])[0]
        rewards[model] = 0
        for start, end in zip(starts,ends):
            unnormalized_model_traj = eval_dataset.unnormalize_states(model_trajectories[start:end+1])
            reward = F110Env.compute_reward_trajectories(unnormalized_model_traj.unsqueeze(0), 
                                                         model_action_trajectories[start:end+1].unsqueeze(0), 
                                                         torch.tensor([len(model_trajectories[start:end+1])]).unsqueeze(0), 
                                                         "reward_progress.json")
            discount_factors = args.discount ** np.arange(reward.shape[1])
            # Calculate the sum of discounted rewards along axis 1
            discounted_sums = np.sum(reward * discount_factors, axis=1)[0]
            if min_reward > discounted_sums:
                min_reward = discounted_sums
            if max_reward < discounted_sums:
                max_reward = discounted_sums
            rewards[model] += discounted_sums
        rewards[model] /= len(starts)
    print(rewards)
    return min_reward, max_reward

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = f"runs_fqe_4_{args.lr}_{args.tau}_{args.weight_decay}_{args.target_reward}",
        algo=args.fqe_model,
        reward_name=args.target_reward,
        dataset="f110-real-stoch-v2",
        target_policy=args.split,
        seed = args.seed,
    )
    print(save_path)
    # append the args.agent to the save path
    save_path = os.path.join(save_path, args.agent) 
    import datetime
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S")
    file_name = str(args.seed) + "_" + time
    writer = SummaryWriter(log_dir= os.path.join(save_path, file_name))
    
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
        normalize_states=True,
        normalize_rewards=False,
        train_only = True,
       # only_agents = [args.agent],
    )

    print("inital states training:", torch.sum(training_dataset.mask_inital))
    actor = Agent().load(name=args.agent, no_print=True)
    get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                                fn_unnormalize_states=training_dataset.unnormalize_states)
    # compute the next actions
    next_actions = get_target_actions(training_dataset.states_next)
    training_dataset.next_actions = next_actions
    #print(training_dataset.next_actions[:20])
    #print(training_dataset.actions[:20])
    #exit()
    ### done ###
    print(args)
    eval_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        only_agents = [args.agent],
        state_mean=training_dataset.state_mean,
        state_std=training_dataset.state_std,
        reward_mean=training_dataset.reward_mean,
        reward_std=training_dataset.reward_std,
    )
    print("inital states eval:", torch.sum(eval_dataset.mask_inital))
    initial_eval_states = eval_dataset.states[np.where(eval_dataset.mask_inital)[0]]
    initial_eval_actions = get_target_actions(initial_eval_states)

    train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)
    # inf_dataloader = get_infinite_iterator(train_loader)
    ###### Sanity Checks ######
    # TODO! how will we do normalization properly????? Trajectory based? Probably not trajectory based, due to later predictions in timesteps
    #min_reward, max_reward = sanity_check(F110Env, eval_dataset)
    # this is appropriate for all of our rewards tbh, but make variable latter
    min_reward = training_dataset.normalize_rewards(0)
    max_reward = training_dataset.normalize_rewards(100) # max(100, max_reward)
    print(torch.std(training_dataset.rewards))
    print("min and max reward")
    print(min_reward, max_reward)
    # normalize min and max_reward according to mean and std
    min_reward = (min_reward - training_dataset.reward_mean) / training_dataset.reward_std
    max_reward = (max_reward - training_dataset.reward_mean) / training_dataset.reward_std
    """
    print(training_dataset.masks.shape)
    print(training_dataset.finished.shape)
    print(training_dataset.states.shape)
    print(eval_dataset.masks.shape)
    print(F110Env.keys)
    # sum of masks and finished
    print(eval_dataset.finished.sum())
    """
    # build the model
    if args.fqe_model == "QFitterL2":
        model = QFitterL2(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], # 256,256,256,256
                min_reward = min_reward,
                max_reward = max_reward,
                average_reward = 30.0,
                critic_lr=1e-7, 
                weight_decay=1e-8,
                tau=0.005,
                #critic_lr=1e-6, 
                #weight_decay=1e-7,
                #tau=0.005,
                discount=args.discount, 
                logger=writer)
    if args.fqe_model == "QFitterLME":
        model = QFitterLME(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                min_reward = min_reward,
                max_reward = max_reward,
                average_reward = 30.0,
                critic_lr=1e-7, 
                weight_decay=1e-8,
                tau=0.005,
                discount=args.discount, 
                logger=writer)
    elif args.fqe_model == "QFitterDD":
        model = QFitterDD(training_dataset.states.shape[1], 
                training_dataset.actions.shape[1], 
                hidden_sizes=[256,256,256,256], 
                num_atoms=101, 
                min_reward=min_reward, 
                max_reward=max_reward, 
                critic_lr=args.lr, #1e-5 -> works
                weight_decay=1e-5,
                tau = args.tau,
                discount=args.discount, 
                logger=writer)

        
    # pbar = tqdm(range(args.update_steps))#, mininterval=5.0)

    import time
    
    

    #for i in pbar:
    i = 0
    pbar = tqdm(total=args.update_steps)
    
    while i < args.update_steps:
        # initalize the datset
        next_actions = get_target_actions(training_dataset.states_next)
        training_dataset.next_actions = next_actions
        train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)

        for step, (states, scans, actions, next_states, next_scans, rewards, masks, sequence_model_names,
            log_prob, next_actions) in enumerate(train_loader):
            if args.train:
                model.set_device("cuda")
                assert not np.isnan(states).any()
                assert not np.isnan(actions).any()
                assert not np.isnan(next_states).any()
                assert not np.isnan(rewards).any()
                assert not np.isnan(masks).any()
                
                # print(masks.sum())
                loss = model.update(states,actions, 
                                    next_states, 
                                    next_actions,
                                    rewards, 
                                    masks)
                #print(end2 - start2)
                writer.add_scalar(f"train/loss_reward", loss, global_step=i)
                writer.add_scalar(f"train/loss", loss, global_step=i)

                if args.save_model and i%10_000 == 0:
                    model.save(save_path, i)

            if i % args.eval_interval == 0:
                #print(save_path)
                #model.load(save_path, i)
                model.set_device("cuda")
                
                if args.eval_only:
                    # save_path = os.path.join("exps",save_path)
                    model.load(save_path, i=190000)
                    model.set_device("cuda")
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions, plot=True)
                    result_dict = dict()
                    result_dict[args.agent] = {"mean": mean.item(),
                                    "std": std.item()}
                    print(result_dict)
                    save_path = os.path.join(save_path, "results")
                    # prepend exps to the save path
                    
                    #if not os.path.exists(save_path):
                    #    os.makedirs(save_path)
                    #with open(os.path.join(save_path, f"{args.target_reward}"), 'w') as file:
                    #    json.dump(result_dict, file)
                    exit()

                #for i in range(10):
                #    print(F110Env.keys[i])
                #    plt.plot(initial_eval_states[:,i])
                #    plt.show()

                if i % 2_000 == 0 and args.fqe_model == "QFitterDD":
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions, plot=True)
                else:
                    mean, std = model.estimate_returns(initial_eval_states, initial_eval_actions)
                # unnomralize the mean and std
                # mean = training_dataset.unnormalize_rewards(mean)
                # std = training_dataset.unnormalize_rewards(std)
                writer.add_scalar(f"eval/mean", mean.item(), global_step=i)
                writer.add_scalar(f"eval/std", std.item(), global_step=i)
                # add the mean to pbar
                pbar.set_postfix({"mean": mean.item(), "std": std.item()})

                #writer.add_scalar(f"train/loss_done", loss_done, global_step=i)
            i += 1
            pbar.update(1)  # Manually update the tqdm progress bar
        
            # Check if we've reached or exceeded the update steps to break the outer loop as well
            if i >= args.update_steps:
                break
        # recreate the train loader


    # update the model

    # evaluate the model

    """
    dynamics_model = build_dynamics_model(args.dynamics_model,
                                          min_state=min_states, 
                                          max_state=max_states)
    ### Define the Model ### 
    model = F110ModelBased(F110Env, behavior_dataset.states.shape[1],
                behavior_dataset.actions.shape[1],
                dynamics_model = dynamics_model,
                hidden_size = [256,256,256,256],
                dt=1/20,
                min_state=min_states,
                max_state=max_states,
                fn_normalize=behavior_dataset.normalize_states,
                fn_unnormalize=behavior_dataset.unnormalize_states,
                use_reward_model=False,
                use_done_model=False,
                obs_keys=behavior_dataset.obs_keys,
                learning_rate=1e-3,
                weight_decay=1e-4,
                target_reward="reward_progress.json",
                logger=writer,)
    # model.load(save_path, filename="model_20000.pth")
    if args.model_checkpoint is not None:
        model.load(save_path, args.model_checkpoint)
    
    pbar = tqdm(range(args.update_steps), mininterval=5.0)

    for i in pbar:
        if args.train:
            model.set_device("cuda")
            (states, scans, actions, next_states, next_scans, rewards, masks, sequence_model_names,
            log_prob) = next(data_iter)
            assert not np.isnan(states).any()
            assert not np.isnan(actions).any()
            assert not np.isnan(next_states).any()
            assert not np.isnan(rewards).any()
            assert not np.isnan(masks).any()

            loss, loss_reward, loss_done = model.update(states, actions, 
                                                        next_states, rewards,
                                                        masks)
            
            writer.add_scalar(f"train/loss_mb", loss, global_step=i)
            writer.add_scalar(f"train/loss_reward", loss_reward, global_step=i)
            writer.add_scalar(f"train/loss_done", loss_done, global_step=i)

        if i % args.eval_interval == 0:
            num_plot = 10
            model.set_device("cpu")
            plot_validation_trajectories(behavior_dataset, model, F110Env, val_indices, num_plot, save_path=save_path, file_name=f"rollouts_{i}")

            if args.save_model:
                model.save(save_path, filename=f"model_{args.seed}_{i}.pth")
            if args.skip_eval: #and (args.update_steps - args.eval_interval > i): # do one evaluation at the end
                continue

            # compute the mse of the model on the test and validation set

            #average_mse_test, std_dev_mse_test = evaluate_model(test_loader, model)
            average_mse_val, std_dev_mse_val = evaluate_model(val_loader, model)
            #writer.add_scalar(f"test/mse_single", average_mse_test, global_step=i)
            writer.add_scalar(f"eval/mse_single", average_mse_val, global_step=i)
            #writer.add_scalar(f"eval/std_mse_test", std_dev_mse_test, global_step=i)
            writer.add_scalar(f"eval/std_mse_val", std_dev_mse_val, global_step=i)


            val_mse = compute_mse_rollouts(behavior_dataset, model, F110Env, val_indices, rollouts=1)
            writer.add_scalar(f"test/mse_trajectory_25", np.mean(val_mse[:25].numpy()), global_step=i)
            writer.add_scalar(f"test/mse_trajectory_50", np.mean(val_mse[:50].numpy()), global_step=i)
            writer.add_scalar(f"test/mse_trajectory_max", np.max(val_mse.numpy()), global_step=i)
            

            #test_mse = compute_mse_rollouts(behavior_dataset, model, F110Env, val_indices)
            #writer.add_scalar(f"test/mse_trajectory_25", np.mean(test_mse[:25].numpy()), global_step=i)
            #writer.add_scalar(f"test/mse_trajectory_50", np.mean(test_mse[:50].numpy()), global_step=i)
            #writer.add_scalar(f"test/mse_trajectory_max", np.max(test_mse.numpy()), global_step=i)
            # print(np.mean(test_mse.numpy()))
                    #all_mse.append(mse)
    """
if __name__ == "__main__":
    main(args)