import os
import json
import numpy as np
import matplotlib.pyplot as plt
# List of DynamicsNetwork names
dynamics_networks = ["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel", "AutoregressiveModel", "EnsemblePDDModel"]  # Replace with your actual network names
max_seed = 25 # Important to change!
base_path = "runs"
def calculate_stats(metrics):
    stats = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics.items()}
    stats["regret@1"] = {"mean": np.mean(metrics["regret@1"]), "std": np.std(metrics["regret@1"])}
    return stats

def plot_stats(stats, metric, title="Mean and STD of metric", save_path=None):
    labels = dynamics_networks #["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel", "AutoregressiveModel"]#list(stats.keys())
    means = [stats[label][metric]['mean'] for label in labels]
    errors = [stats[label][metric]['std'] for label in labels]

    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, means, width, label='Mean', yerr=errors, capsize=5)
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    #ax.legend()
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
def read_metrics_from_json(network_name, reward_name="reward_progress"):
    network_path = os.path.join(base_path, network_name, "f110-real-stoch-v2/250/off-policy")
    results = {"spearman_corr": [], "abs": [], "regret@1": []}
    file_not_found_counter = 0 # Counter for files not found
    if os.path.exists(network_path):
        seeds = [d for d in os.listdir(network_path) if os.path.isdir(os.path.join(network_path, d))]
        for seed in seeds:
            if int(seed) > max_seed:
                continue
            json_path = os.path.join(network_path, seed, "results", f"reward_{reward_name}._metrics.json")
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    print(json_path)
                    data = json.load(f)
                    if 'spearman_corr' in data and 'abs' in data:
                        results['spearman_corr'].append(data['spearman_corr'])
                        results['abs'].append(data['abs'])
            else:
                print(f"File not found: {json_path}")
                file_not_found_counter += 1
            # now read the specific value and compute the regret@1
            # load the ground truth reward
            ground_truth_rewards_folder = f"/home/fabian/msc/f110_dope/ws_release/experiments/runs3/Groundtruth/gt_{reward_name}.json"
            with open(ground_truth_rewards_folder, "r") as f:
                ground_truth_rewards = json.load(f)
            # get the max from the ground truth rewards
            max_reward = max([ground_truth_rewards[agent]['mean'] for agent in ground_truth_rewards])
            json_path = os.path.join(network_path, seed, "results", f"{reward_name}.json")
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    print(json_path)
                    data = json.load(f)
                    max_reward_agent = max([data[agent]['mean'] for agent in data])
                    print(max_reward - max_reward_agent)
                    results['regret@1'].append(abs(max_reward - max_reward_agent))
            else:
                print(f"File not found: {json_path}")
                file_not_found_counter += 1
            print(f"Regret@1: {results['regret@1']}")
    else:
        print(f"Path does not exist: {network_path}")
    
    return results, file_not_found_counter

# Iterate through each dynamics network and print results
for reward in ["reward_progress", "reward_checkpoint", "reward_lifetime", "reward_min_act"]:
    all_stats = {}
    for network_name in dynamics_networks:
        metrics, files_not_founds = read_metrics_from_json(network_name, reward_name=reward)
        print(metrics)

        stats = calculate_stats(metrics)
        all_stats[network_name] = stats
        print(f"Results for {network_name}:")
        print(f"Spearman Correlation: {metrics['spearman_corr']}")
        print(f"Absolute values: {metrics['abs']}\n")
        print(f"Regret@1: {metrics['regret@1']}\n")

    print(all_stats)
    plot_stats(all_stats, "spearman_corr", title=f"Spearman Correlation {reward}", save_path=f"spearman_corr_{reward}.png")
    plot_stats(all_stats, "abs", title=f"Absolute error {reward}", save_path=f"abs_{reward}.png")
    plot_stats(all_stats, "regret@1", title=f"Regret@1 {reward}", save_path=f"regret@1_{reward}.png")
    # print files not found with warning if > 0
    if files_not_founds > 0:
        print(f"Warning: {files_not_founds} files not found")
    else:
        print(f"All files found for {reward}")