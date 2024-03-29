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
parser.add_argument('--output_folder', type=str, default="runs3", help="where to save the ground truth rewards")
parser.add_argument('--zarr_path', type=str, default=None, help="path to the zarr file if not using default")
args = parser.parse_args()
from create_plots import plot_bars_from_dicts

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
            render_mode="human")
    )
    if args.zarr_path is not None:
        dataset = F110Env.get_dataset(zarr_path = args.zarr_path,eval_only=True)
    else:
        dataset = F110Env.get_dataset(eval_only=True)

    ground_truth_rewards = {}
    trajectories, action_trajectories, terminations, model_names = F110Env.compute_trajectories(dataset["observations"],
                                                                                                dataset["actions"],
                                                                                                dataset["terminals"],
                                                                                                dataset["timeouts"], 
                                                                                                dataset["model_name"])

    #(829, 250, 9)
    print(trajectories.shape)
    #exit()
    all = []
    import matplotlib.pyplot as plt
    # List of DynamicsNetwork names
    import seaborn as sns
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
    })
    for model in np.unique(dataset["model_name"]):#tqdm(F110Env.eval_agents):
        model_trajectories = trajectories[model_names == model]
        model_action_trajectories = action_trajectories[model_names == model]
        model_terminations = terminations[model_names == model]
        #print(len(model_trajectories))
        #print(model
        reward = F110Env.compute_reward_trajectories(model_trajectories, model_action_trajectories, model_terminations, args.target_reward)
        
        discount_factors = args.discount ** np.arange(trajectories.shape[1])
        # Calculate the sum of discounted rewards along axis 1
        discounted_sums = np.sum(reward * discount_factors, axis=1)
        # print(f"Model: {model}, Reward: {reward}")
        all.append(discounted_sums)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(all):
        plt.hist(data, bins=30, alpha=0.5, label=f'Model {i+1}')
    # fix the y axis between 0 and 100
    plt.xlim(0, 100)
    plt.ylim(0,30)
    plt.xticks(fontsize=20)
    plt.xlabel('Discounted Sums', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.title('Histogram of Discounted Sums Across Models', fontsize=22)
    #plt.legend()
    # plt.grid(True)
    # save
    plt.show()
    plt.savefig(f"plots/a_histogram_{args.target_reward}.pdf")
    

    # write the ground truth rewards to a file
    #with open(f"Groundtruth/gt_{args.target_reward}", "w") as f:
    #    json.dump(ground_truth_rewards, f)
    #plot_bars_from_dicts([ground_truth_rewards], ["Ground truth"], "Mean discounted reward",plot=True)
    #print(ground_truth_rewards)


if __name__ == "__main__":
    main(args)
    