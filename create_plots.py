import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np

import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
from ope_methods.dataset import create_save_dir



from f110_orl_dataset.plot_reward import calculate_discounted_reward, plot_rewards
from scipy.stats import linregress
from scipy.stats import spearmanr

def plot_rewards(ground_truth, computed, title = "Ground Truth vs Computed Rewards", save_path = None, plot=False, method ='Computed Mean Reward' ):
    # Combine the ground truth and computed dictionaries
    combined_data = []
    for agent_name, gt_values in ground_truth.items():
        if agent_name in computed:
            combined_data.append((gt_values['mean'], computed[agent_name]['mean'], agent_name))

    # Sort by ground truth mean magnitude
    combined_data.sort(key=lambda x: x[0])

    # Separate data for plotting
    gt_means = [x[0] for x in combined_data]
    computed_means = [x[1] for x in combined_data]
    labels = [x[2] for x in combined_data]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_means, computed_means, color='blue')
    #for i, label in enumerate(labels):
    #    plt.text(gt_means[i], computed_means[i], label, fontsize=9)
    
    # Fit a line
    slope, intercept, r_value, p_value, std_err = linregress(gt_means, gt_means)
    space = np.arange(0,max(gt_means)+3)
    line = slope * np.array(space) + intercept
    plt.plot(space, line, 'r') #label=f'y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.2f}')
    # fix the x and y axis to the ground truth
    plt.xlim(5, max(gt_means)+1)
    plt.ylim(5, max(max(gt_means), max(computed_means))+1)
    plt.xlabel('Ground Truth Mean Reward')
    plt.ylabel(method)
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    if plot:
        plt.show()
    else:
        plt.close()
def plot_bars_from_dicts(dicts_list, dict_names, add_title="", save_path=None, plot=False):
    """
    Plots bar charts for given dictionaries, with data categorized by sub-keys found in the first dictionary.

    :param dicts_list: List of dictionaries with sub-key structured data to plot.
    :param dict_names: Names corresponding to each dictionary, used as labels.
    :param add_title: Additional title for the plot.
    """
    assert len(dicts_list) == len(dict_names), "Each dictionary must have a corresponding name."

    sub_keys = list(dicts_list[0].keys())  # Sub-keys from the first dictionary
    num_dicts = len(dicts_list)
    bar_width = 0.15  # Adjust for spacing
    colors = plt.cm.viridis(np.linspace(0, 1, len(sub_keys)))
    num_bars = len(sub_keys)
    fig, ax = plt.subplots(figsize=(12, 7))
    r = np.arange(len(dict_names)) *3
    print(r)
    for i, dict_name in enumerate(dict_names):
        for j, sub_key in enumerate(sub_keys):
            x_coord = r[i] + j * bar_width - (bar_width * (num_bars - 1) / 2)
            mean = dicts_list[i][sub_key]['mean']
            std = dicts_list[i][sub_key]['std']
            if i == 0:
                ax.bar(x_coord, mean, yerr=std, width=bar_width, label=sub_key, alpha=0.8, capsize=7, color=colors[j])
            else:
                ax.bar(x_coord, mean, yerr=std, width=bar_width, alpha=0.8, capsize=7, color=colors[j])

    ax.set_xlabel('Sub-keys')
    ax.set_ylabel('Values')
    ax.set_title(f"{add_title}")
    ax.set_xticks(r)
    ax.set_xticklabels(dict_names)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Agents")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
   
    if plot:
        plt.show()
    else:
        plt.close()


def main(args):
    # load the dataset:
    """
    F110Env = gym.make("f110-real-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config="reward_progress.json",
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
            render_mode="human")
    )
    dataset = F110Env.get_dataset()
    #discounted = calculate_discounted_reward()
    print(dataset["actions"].shape)
    trajectories, action_trajectories, terminations, model_names = F110Env.compute_trajectories(dataset["observations"],dataset["actions"], dataset["terminals"], dataset["timeouts"], dataset["model_name"] )
    """
    # read in computed rewards from json file
    compute_rewards = {}
    save_path = create_save_dir(
        experiment_directory = "runs",
        algo=args.dynamics_model,
        reward_name=args.target_reward,
        dataset="f110-real-stoch-v2",
        target_policy=args.split,
        seed = args.seed,
    )
    #before save path prepend the current path
    save_path = os.path.join(os.getcwd(), save_path)

    with open(f"{save_path}/results/{args.target_reward}", "r") as f:
        compute_rewards = json.load(f)
    #target_rewards_folder = "/home/fabian/msc/f110_dope/ws_release/experiments/runs3/Simulation/sim_reward_progress.json"
    #with open(target_rewards_folder, "r") as f:
    #    compute_rewards = json.load(f)

    ground_truth_rewards_folder = f"/home/fabian/msc/f110_dope/ws_release/experiments/runs3/Groundtruth/gt_{args.target_reward}"
    with open(ground_truth_rewards_folder, "r") as f:
        ground_truth_rewards = json.load(f)
    # now we plot 1) spearman correlation 2) mean and std of the discounted rewards
    # 3) MSE
    gt_means = [ground_truth_rewards[agent]['mean'] for agent in ground_truth_rewards if agent in compute_rewards]
    computed_means = [compute_rewards[agent]['mean'] for agent in ground_truth_rewards if agent in compute_rewards]

    # Compute Spearman correlation coefficient
    spearman_corr, p_value = spearmanr(gt_means, computed_means)
    #print(f"Spearman Correlation Coefficient: {spearman_corr}, p-value: {p_value}")
    plot_rewards(ground_truth_rewards, compute_rewards, title=f"Reward: {args.target_reward}, {args.dynamics_model}, spearman_corr: {spearman_corr:.2f}, p-value: {p_value:.5f}",
                 save_path=f"{save_path}/results/reward_{args.target_reward[:-4]}_spearman_corr.png", plot=args.plot)
    # compute the mse between gt and computed rewards
    abs = np.mean(np.abs((np.array(gt_means) - np.array(computed_means))))
    print(abs)
    abs_std = np.std((np.array(gt_means) - np.array(computed_means)))
    #print(f"MSE: {mse}")
    # plot the bar-chart with st deviations
    gt_stds = [ground_truth_rewards[agent]['std'] for agent in ground_truth_rewards if agent in compute_rewards]
    computed_stds = [compute_rewards[agent]['std'] for agent in ground_truth_rewards if agent in compute_rewards]


    # also do the bar chart plots

    plot_bars_from_dicts([ground_truth_rewards, compute_rewards], dict_names=["Ground Truth", "Computed"], add_title=f"Reward: {args.target_reward}, {args.dynamics_model} , Absolute Error: {abs:.2f},",#",
                         save_path=f"{save_path}/results/reward_{args.target_reward[:-4]}_bars.png", plot=args.plot)

    # save a dictionary with the mse and spearman corr
    results_dict = {"spearman_corr": spearman_corr, "p_value": p_value, "abs": abs, "abs_std": abs_std}
    with open(f"{save_path}/results/reward_{args.target_reward[:-4]}_metrics.json", "w") as f:
        json.dump(results_dict, f)
    print(results_dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model based approaches')

    # target reward
    parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
    parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
    parser.add_argument('--policy_folder', type=str, default="runs3", help="policy folder")
    parser.add_argument('--dynamics_model', type=str, default="dynamics_model", help="dynamics model")
    parser.add_argument('--split', type=str, default="off-policy", help="split")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--plot', action="store_true", help="plot the results")
    args = parser.parse_args()
    main(args)
    