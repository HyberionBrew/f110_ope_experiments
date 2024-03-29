import numpy as np
import torch
import ope_methods
import f110_gym
import f110_orl_dataset
from ope_methods.dataset import F110Dataset
import gymnasium as gym
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Subset
import os
from ope_methods.model_based import F110ModelBased
import json
from create_plots import plot_bars_from_dicts

parser = argparse.ArgumentParser(description='Evaluation of Model-based OPE methods')
parser.add_argument('--dynamics_model', type=str, default="DeltaDynamicsModel", help="dynamics model")
parser.add_argument('--split', type=str, default="on-policy", help="split")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--model_checkpoint', type=str, default="model.pt", help="model checkpoint")

parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--plot', action='store_true', help="Enable plotting")

parser.add_argument('--compute_gt', action='store_true', help="Compute ground truth")
parser.add_argument('--target_reward', type=str, nargs='+', default=["reward_progress.json"], help="target reward")

parser.add_argument('--rollouts_per_initial_state', type=int, default=5, help="rollouts per initial state")
parser.add_argument('--save', action='store_true', help="Save the results")
args = parser.parse_args()

def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    save_path = ope_methods.dataset.create_save_dir(
        experiment_directory = "runs_mb",
        algo=args.dynamics_model,
        reward_name="reward_progress",
        dataset="f110-real-stoch-v2",
        target_policy=args.split,
        seed=args.seed,
    )
    
    F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config=None,
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = False,
        include_progress=False,
        set_previous_step_terminals=0,
        use_compute_termination=True,
        remove_cons_terminals=True,
        **dict(name="f110-real-stoch-v2",
            config = dict(map="Infsaal3", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human_fast")
    )

    behavior_dataset = F110Dataset(
        F110Env,
        normalize_states=True,
        normalize_rewards=False,
        # sequence_length=horizon,
        # only_agents= [F110Env.eval_agents], # include all agents in the dataset
    )
    #load the train, val and test indices
    #train_indices = np.load(os.path.join(save_path, "train_indices.npy"))
    val_indices = np.load(os.path.join("/home/fabian/msc/f110_dope/ws_release/experiments/runs_mb", "val_indices.npy"))
    #test_indices = np.load(os.path.join(save_path, "test_indices.npy"))
    #train_subset = Subset(behavior_dataset, train_indices)
    #test_subset = Subset(behavior_dataset, test_indices)
    #val_subset = Subset(behavior_dataset, val_indices)
    #test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    #val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    # build dynamics model
    from ope_methods.model_based import build_dynamics_model


    min_states = behavior_dataset.states.min(axis=0)[0]
    max_states = behavior_dataset.states.max(axis=0)[0]
    min_states[:2] = min_states[:2] - 2.0
    max_states[:2] = max_states[:2] + 2.0
    dynamics_model = build_dynamics_model(args.dynamics_model, 
                                          min_state=min_states, 
                                          max_state=max_states)
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
            logger=None,)
    model.load(save_path, args.model_checkpoint)

    # for each agent in the dataset, estimate the returns from its specific starting positions
    # build the trajectories from the dataset
    print("Number of valdation transitions:", val_indices.shape)
    print(args.target_reward)
    val_dataset = behavior_dataset.get_only_indices(val_indices)
    print(len(val_dataset))
    # plot the val dataset
    # build trajectories
    starts = np.where(np.roll(val_dataset.finished, 1) == 1)[0]
    ends = np.where(val_dataset.finished == 1)[0]
    # find horizon
    horizon = np.max(ends - starts)
    trajectories = np.zeros((len(starts),250, val_dataset.states.shape[1]))
    terminations = np.zeros((len(starts),))
    truncations = np.zeros((len(starts),))
    # empty strings of size len(starts)
    model_names = np.empty((len(starts),), dtype=object)
    #print(val_dataset.finished)

    # build trajectories
    for i, (start, end) in enumerate(zip(starts, ends)):
        #print(start,end)
        #print(val_dataset.states[start:end+1].shape)
        trajectories[i, 0:end - start+ 1] = val_dataset.states[start:end+1]
        term = np.where(1.0 - val_dataset.masks[start:end+1])[0]
        if len(term )== 0:
            term = [horizon+1]

        terminations[i] = int(term[0])
        truncations[i] = np.where(val_dataset.finished[start:end+1])[0][0]
        model_names[i] = val_dataset.model_names[start]
        #trajectories.append(val_dataset.states[start:end+1])
        #terminations.append(np.where(val_dataset.masks[start:end+1])[0])
        #truncations.append(np.where(val_dataset.finished[start:end+1])[0])
        #model_names.append(val_dataset.model_names[start])
    model_names = np.array(model_names)
    # replace nones with ""
    model_names[model_names == None] = ""
    # F110Env.plot_trajectories(trajectories,model_names, terminations, truncations)
    # do rollouts and later compute the rewards
    from f110_agents.agent import Agent
    from functools import partial
    result_dict = {reward_function: {} for reward_function in args.target_reward}
    for agent_name in np.unique(model_names):
        if agent_name is None:
            continue
        try:
            actor = Agent().load(name=agent_name)
        except FileNotFoundError:
            print(f"Agent {agent_name} not found")
            continue
        get_target_actions = partial(F110Env.get_target_actions, actor=actor, 
                            fn_unnormalize_states=behavior_dataset.unnormalize_states)
        curr_trajectories = np.where(model_names == agent_name)[0]
        inital_states = trajectories[curr_trajectories,0,:]
        # copy inital states 5 times
        inital_states = np.repeat(inital_states, args.rollouts_per_initial_state, axis=0)
        # unsqueeze the initial states at axis = 1
        
        inital_states = torch.tensor(np.expand_dims(inital_states, axis=1),dtype=torch.float32)

        states_rollout, actions_rollout = model.rollout(inital_states,
                                                        get_target_action= get_target_actions,
                                                        horizon=horizon+1)
        
        states_unnormalized = behavior_dataset.unnormalize_states(states_rollout)
        terminations_rollout = model.termination_model(states_unnormalized, 
                                               map_path = None)
                                               #"/home/fabian/msc/f110_dope/ws_release/f1tenth_gym/gym/f110_gym/maps/Infsaal3/Infsaal3_centerline.csv")
        truncations_rollout = np.ones((len(curr_trajectories),)) * (horizon)
        # print the first 10 unnormalized states
        dict_rewards = {}
        for target_reward in args.target_reward:
            #print(actions_rollout.shape)
            #print(states_unnormalized.shape)
            dict_rewards[target_reward] = F110Env.compute_reward_trajectories(states_unnormalized,
                                            actions_rollout, 
                                            terminations_rollout, 
                                            reward_config=target_reward)
        # plot the first 3 reward
        """
        import matplotlib.pyplot as plt
        for t in args.target_reward:
            print(target_reward)
            print(dict_rewards[t].shape)
            plot = 10
            F110Env.plot_trajectories(states_unnormalized[:plot],np.array([agent_name]*plot), 
                                      terminations_rollout[:plot], 
                                      title = f"Rollouts {agent_name}")
            for i in range(10):
                plt.plot(dict_rewards[t][i,:])

            plt.title(f"Reward {t}")
            plt.show()
        """
        discount_factors = args.discount ** np.arange(horizon + 1)
        # unnormed_trajectories = behavior_dataset.unnormalize_states(torch.tensor(trajectories[curr_trajectories]))
        if args.compute_gt:
            raise DeprecationWarning("This part of the code is deprecated")
           
            #print(unnormed_trajectories[0:2,:3])
            #import matplotlib.pyplot as plt
            #to_plot = 3
            #F110Env.plot_trajectories(states_unnormalized[:to_plot].squeeze().detach().numpy(),model_names[curr_trajectories][:to_plot], terminations_rollout[:to_plot], truncations_rollout[:to_plot], title = f"Rollouts {agent_name}")
            #if args.compute_gt:
            #    F110Env.plot_trajectories(unnormed_trajectories[:to_plot],model_names[curr_trajectories][:to_plot], terminations[curr_trajectories][:to_plot], truncations[curr_trajectories][:to_plot], title=f"Ground Truth {agent_name}")
  
            reward_gt = F110Env.compute_reward_trajectories(unnormed_trajectories,
                                            actions_rollout, 
                                            terminations[curr_trajectories], 
                                reward_config= args.target_reward)
            # plot the reward of the ground truth vs the rollouts
            """
            for i in range(3):
                # calculate the gt discounted reward
                discount_factors = args.discount ** np.arange(horizon + 1)
                discounted_sum_gt = np.sum(reward_gt[i] * discount_factors)
                print(discounted_sum_gt)
                # same for the rollouts
                discounted_sum_rollout = np.sum(reward[i] * discount_factors)
                print(discounted_sum_rollout)

                plt.plot(reward_gt[i], color = "red")
                plt.plot(reward[i], color= "blue")
                plt.legend(["Ground Truth", "Rollouts"])
                plt.show()
            """
            discounted_sum_gt = np.sum(reward_gt *discount_factors, axis=1)
            mean_reward_gt = np.mean(discounted_sum_gt)
            std_reward_gt = np.std(discounted_sum_gt)
            print("Mean reward gt:", mean_reward_gt)
            print("Std reward gt:", std_reward_gt)
            print("Max reward gt:", np.max(discounted_sum_gt))
            print("Min reward gt:", np.min(discounted_sum_gt))
            print(discounted_sum_gt )
        # Create a discount factor array of shape (250,)
        
        # Calculate the sum of discounted rewards along axis 1
        for target_reward in args.target_reward:
            discounted_sums = np.sum(dict_rewards[target_reward] * discount_factors, axis=1)
            mean_reward_rollouts = np.mean(discounted_sums)
            std_reward_rollouts = np.std(discounted_sums)
            print("Mean reward rollouts:", mean_reward_rollouts)
            print("Std reward rollouts:", std_reward_rollouts)
            print("Max reward rollouts:", np.max(discounted_sums))
            print("Min reward rollouts:", np.min(discounted_sums))


            result_dict[target_reward][agent_name] = {"mean": mean_reward_rollouts,
                                    "std": std_reward_rollouts}
        # print("Mean reward rollouts:", mean_reward_rollouts)
        #print("Mean reward gt:", mean_reward_gt)
        if args.plot:
            if False:
                print(discounted_sums)
                print(states_unnormalized.shape)
                print(model_names[curr_trajectories].shape)
                F110Env.plot_trajectories(states_unnormalized,
                                        model_names[curr_trajectories],
                                        terminations_rollout,
                                        title = f"Rollouts {agent_name}")
                #if args.compute_gt:
            #    F110Env.plot_trajectories(unnormed_trajectories,model_names[curr_trajectories], terminations[curr_trajectories], truncations[curr_trajectories], title=f"Ground Truth {agent_name}")
    print(result_dict)
    # write the result dict to a file
    # create a new directory results
    # add "results" to save path
    save_path = os.path.join(save_path, "results")
    for target_reward in args.target_reward:
        if args.plot:
            plot_bars_from_dicts([result_dict[target_reward]], ["Rollouts"], f"Mean discounted reward {target_reward}",plot=True)
        if args.save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, f"{target_reward}"), 'w') as file:
                json.dump(result_dict[target_reward], file)
            
            
if __name__ == "__main__":
    main(args)
    