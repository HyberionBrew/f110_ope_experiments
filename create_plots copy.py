import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np

import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train model based approaches')

# target reward
parser.add_argument('--target_reward', type=str, default="reward_progress.json", help="target reward")
parser.add_argument('--discount', type=float, default=0.99, help="discount factor")
parser.add_argument('--policy_folder', type=str, default="runs3", help="policy folder")
args = parser.parse_args()

from f110_orl_dataset.plot_reward import calculate_discounted_reward, plot_rewards
from scipy.stats import linregress
from scipy.stats import spearmanr

def plot_rewards(ground_truth, computed, title = "Ground Truth vs Computed Rewards", save_path = None):
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
    plt.ylim(5, max(gt_means)+1)
    plt.xlabel('Ground Truth Mean Reward')
    plt.ylabel('Computed Mean Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main(args):
    # load the dataset:
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
    # read in computed rewards from json file
    compute_rewards = {}
    with open(f"{args.policy_folder}/{args.target_reward}", "r") as f:
        compute_rewards = json.load(f)
    
    ground_truth_rewards = {}
    print(model_names[0])
    print(type(model_names[0]))
    print(type(str(model_names[0])))
    print((np.where(model_names is str('StochasticContinuousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5'))))
    print(np.unique(model_names))
    """
    for model in tqdm(F110Env.eval_agents):
        model_trajectories = trajectories[model_names == model]
        model_action_trajectories = action_trajectories[model_names == model]
        model_terminations = terminations[model_names == model]
        # print(model_trajectories)
        reward = F110Env.compute_reward_trajectories(model_trajectories, model_action_trajectories, model_terminations, args.target_reward)
        discount_factors = args.discount ** np.arange(trajectories.shape[1])
        # Calculate the sum of discounted rewards along axis 1
        discounted_sums = np.sum(reward * discount_factors, axis=1)
        # print(f"Model: {model}, Reward: {reward}")
        ground_truth_rewards[model] = {
            "mean": np.mean(discounted_sums),
            "std": np.std(discounted_sums)
        }
    """
        # break
    print(ground_truth_rewards)
    # now we plot 1) spearman correlation 2) mean and std of the discounted rewards
    # 3) MSE
    gt_means = [ground_truth_rewards[agent]['mean'] for agent in ground_truth_rewards if agent in compute_rewards]
    computed_means = [compute_rewards[agent]['mean'] for agent in ground_truth_rewards if agent in compute_rewards]

    # Compute Spearman correlation coefficient
    spearman_corr, p_value = spearmanr(gt_means, computed_means)
    print(f"Spearman Correlation Coefficient: {spearman_corr}, p-value: {p_value}")
    plot_rewards(ground_truth_rewards, compute_rewards, title=f"Reward: progress, ProbsDeltaDynamics, spearman_corr: {spearman_corr:.2f}, p-value: {p_value:.5f}",
                 save_path=f"{args.policy_folder}/reward_progress_spearman_corr.png")


    
if __name__ == "__main__":
    main(args)
    