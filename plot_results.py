import os
import json
import numpy as np
import matplotlib.pyplot as plt
# List of DynamicsNetwork names
dynamics_networks = ["DeltaDynamicsModel", "ProbDynamicsModel", "SimpleDynamicsModel","ProbsDeltaDynamicsModel"]  # Replace with your actual network names

base_path = "runs3"
def calculate_stats(metrics):
    stats = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics.items()}
    return stats

def plot_stats(stats, metric, title="Mean and STD of metric"):
    labels = ["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel"]#list(stats.keys())
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
    plt.show()
def read_metrics_from_json(network_name):
    network_path = os.path.join(base_path, network_name, "f110-real-stoch-v2/250/off-policy")
    results = {"spearman_corr": [], "abs": []}

    if os.path.exists(network_path):
        seeds = [d for d in os.listdir(network_path) if os.path.isdir(os.path.join(network_path, d))]
        for seed in seeds:
            json_path = os.path.join(network_path, seed, "results", "reward_reward_progress._metrics.json")
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'spearman_corr' in data and 'abs' in data:
                        results['spearman_corr'].append(data['spearman_corr'])
                        results['abs'].append(data['abs'])
            else:
                print(f"File not found: {json_path}")
    else:
        print(f"Path does not exist: {network_path}")
    
    return results

# Iterate through each dynamics network and print results
all_stats = {}
for network_name in dynamics_networks:
    metrics = read_metrics_from_json(network_name)
    stats = calculate_stats(metrics)
    all_stats[network_name] = stats
    print(f"Results for {network_name}:")
    print(f"Spearman Correlation: {metrics['spearman_corr']}")
    print(f"Absolute values: {metrics['abs']}\n")
print(all_stats)
plot_stats(all_stats, "spearman_corr", title="Spearman Correlation")
plot_stats(all_stats, "abs", title="Absolute error")