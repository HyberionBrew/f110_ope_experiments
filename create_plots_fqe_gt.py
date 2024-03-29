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
from create_plots import plot_bars_from_dicts, plot_rewards
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})

def plot_rewards_new(ground_truth, computed, title = "Ground Truth vs Computed Rewards", save_path = None, plot=False, method ='Computed Mean Reward' ):
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
    sns.set_theme(style="whitegrid")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=gt_means, y=computed_means, color='blue', s=100, alpha=0.7)
    
    # Ideal line (diagonal where ground truth equals computed rewards)
    max_val = max(max(gt_means), max(computed_means))+3
    plt.plot([0, max_val], [0, max_val], 'r', linestyle='--', label='Ideal Fit')

    # Styling
    plt.xlabel('Real-world Return', fontsize=20)
    plt.ylabel(method, fontsize=20)
    plt.title(title, fontsize=22)
    plt.xlim(0, max(gt_means)+3)
    plt.ylim(0, max_val)
    # increase the font size of the ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if save_path:
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
    target_path = f"/home/fabian/msc/f110_dope/ws_release/experiments/exps/runs_fqe_4_0.0001_0.005_0.0/QFitterDD/f110-real-stoch-v2/250/on-policy/{args.seed}"
    target_rewards_folder = os.path.join(target_path, args.target_reward) #args.target_reward}"
    # join(target_path, args.target_reward) #args.target_reward}"
    with open(target_rewards_folder, "r") as f:
        compute_rewards = json.load(f)

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
    reward_string = args.target_reward[7:-5]
    abs = np.mean(np.abs((np.array(gt_means) - np.array(computed_means))))

    abs_std = np.std((np.array(gt_means) - np.array(computed_means)))
    plot_rewards_new(ground_truth_rewards, compute_rewards, title=f"{reward_string}-reward\n rank-corr: {spearman_corr:.2f} (p={p_value:.3f}), mean-absolute-error: {abs:.1f}",#Reward: {args.target_reward}, {args.dynamics_model}, ",
                 save_path=f"{target_path}/{args.target_reward[:-5]}_spearman_corr.pdf", plot=args.plot, method="FQE Estimate")
    # compute the mse between gt and computed rewards

    #print(f"MSE: {mse}")
    # plot the bar-chart with st deviations
    gt_stds = [ground_truth_rewards[agent]['std'] for agent in ground_truth_rewards if agent in compute_rewards]
    computed_stds = [compute_rewards[agent]['std'] for agent in ground_truth_rewards if agent in compute_rewards]


    # also do the bar chart plots

    plot_bars_from_dicts([ground_truth_rewards, compute_rewards], dict_names=["Ground Truth", "Computed"], add_title=f"Reward: {args.target_reward}, {args.dynamics_model} , Absolute Error: {abs:.2f},",#",
                         save_path=f"{target_path}/gt_fqe_{args.target_reward[:-5]}_bars.png", plot=args.plot)

    # save a dictionary with the mse and spearman corr
    results_dict = {"spearman_corr": spearman_corr, "p_value": p_value, "abs": abs, "abs_std": abs_std}
    with open(f"{target_path}/gt_fqe_{args.target_reward[:-4]}_metrics.json", "w") as f:
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
    