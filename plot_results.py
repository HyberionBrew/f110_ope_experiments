import os
import json
import numpy as np
import matplotlib.pyplot as plt
# List of DynamicsNetwork names
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})

dynamics_networks = ["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel","AutoregressiveModel", "AutoregressiveDeltaModel"] #"EnsemblePDDModel", "EnsembleSDModel", "EnsembleARModel", "EnsembleARDModel"] #,"ProbsDeltaDynamicsModel", "AutoregressiveModel", "EnsemblePDDModel", "NormImportanceSampling"]  # Replace with your actual network names
# take all folder names from:
#dir_iw = "/home/fabian/msc/f110_dope/ws_release/experiments/runs_iw3"
#dynamics_networks= [d for d in os.listdir(dir_iw) if os.path.isdir(os.path.join(dir_iw, d))]
dynamics_networks = ['iw_action_step_wis_termination_mean', 'iw_action_simple_step_is_zero',  
                     'iw_action_simple_is_mean',
                      'iw_action_step_wis_mean', 'iw_action_simple_step_is_mean', 
                      'iw_action_cobs_wis_zero', 'iw_action_step_wis_zero']
print(dynamics_networks)
#dynamics_networks = ["iw_action_cobs_wis_mean", "iw_action_step_wis_termination_mean", "iw_action_simple_step_is_mean"]
max_seed = 1 # Important to change!
#base_path = "runs_mb"
base_path= "runs_iw3"
def calculate_stats(metrics):
    stats = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics.items()}
    stats["regret@1"] = {"mean": np.mean(metrics["regret@1"]), "std": np.std(metrics["regret@1"])}
    return stats

def plot_stats_comparison(stats1, stats2, metric, title="Comparison of Mean and STD of Metric", save_path=None):
    # labels = dynamics_networks  # List of labels assumed to be defined elsewhere
    labels_1 = list(stats1.keys())
    labels_2 = list(stats2.keys())
    means1 = [stats1[label][metric]['mean'] for label in labels_1]
    errors1 = [stats1[label][metric]['std'] for label in labels_1]
    means2 = [stats2[label][metric]['mean'] for label in labels_2]
    errors2 = [stats2[label][metric]['std'] for label in labels_2]
    
    # Simplifying label names by removing "Model"
    labels = [label.replace("Model", "") for label in labels_1]
    
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35  # Width of the bars
    
    # Plotting
    plt.bar(x - width/2, means1, width, label='Stats1 Mean', yerr=errors1, capsize=5, color='skyblue')
    plt.bar(x + width/2, means2, width, label='Stats2 Mean', yerr=errors2, capsize=5, color='orange')
    
    # plt.xlabel('Dynamics Network', fontsize=20)
    if metric == "spearman_corr":
        plt.ylabel("Spearman Correlation", fontsize=20)
    elif metric == "abs":
        plt.ylabel("mean-absolute error", fontsize=20)
    
    plt.title(title, fontsize=22)
    plt.xticks(x, labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y')
    
    # Adjusting the y-axis limits based on the metric
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    elif metric == "abs":
        plt.ylim(0, 50)
    
    # overwrite legend with 1) Naive, 2) Ensemble
    plt.legend(["Naive", "Ensemble"], fontsize=20)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_stats(stats, metric, title="Mean and STD of metric", save_path=None):
    labels = dynamics_networks #["SimpleDynamicsModel","ProbDynamicsModel","DeltaDynamicsModel","ProbsDeltaDynamicsModel", "AutoregressiveModel"]#list(stats.keys())
    means = [stats[label][metric]['mean'] for label in labels]
    errors = [stats[label][metric]['std'] for label in labels]
    # from each label remove the Model part
    labels = [label.replace("Model", "") for label in labels]
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    #fig, ax = plt.subplots()
    plt.bar(x - width/2, means, width, label='Mean', yerr=errors, capsize=5)
    plt.xlabel('Dynamics Network', fontsize=20)
    if metric == "spearman_corr":
        plt.ylabel("Spearman Correlation", fontsize=20)
    if metric == "abs":
        plt.ylabel("mean-absolute error", fontsize=20)
    #plt.ylabel(metric, fontsize=20)
    plt.title(title, fontsize=22)
    plt.xticks(x, labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    # add grid x direction
    plt.grid(axis='y')
    # fix the y axis between -0.3 and 1
    if metric == "spearman_corr":
        plt.ylim(-0.5, 1)
    if metric == "abs":
        plt.ylim(0, 50)
    #ax.set_ylabel('Scores')
    #ax.set_title(title)
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels, rotation=45)
    #ax.legend()
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
def read_metrics_from_json(network_name, reward_name="reward_progress"):
    network_path = os.path.join(base_path, network_name, "f110-real-stoch-v2/250/off-policy")
    #network_path = os.path.join("/home/fabian/msc/f110_dope/ws_release/experiments/runs_iw3", network_name, "f110-real-stoch-v2/250/off-policy")
    results = {"spearman_corr": [], "abs": [], "regret@1": []}
    file_not_found_counter = 0 # Counter for files not found
    if os.path.exists(network_path):
        seeds = [d for d in os.listdir(network_path) if os.path.isdir(os.path.join(network_path, d))]
        for seed in seeds:
            if int(seed) > max_seed:
                continue
            json_path = os.path.join(network_path, seed, "results", f"reward_{reward_name}._metrics.json")
            if os.path.isfile(json_path):
                if network_name == "iw":
                    print("??")
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
    plot_stats(all_stats, "spearman_corr", title=f" {reward[7:]}-reward \n Spearman Correlation", save_path=f"plots/spearman_corr_{reward}.pdf")
    plot_stats(all_stats, "abs", title=f" {reward[7:]}-reward \nAbsolute error", save_path=f"plots/abs_{reward}.pdf")
    plot_stats(all_stats, "regret@1", title=f" {reward[7:]}-reward \n Regret@1", save_path=f"plots/regret@1_{reward}.pdf")
    models = ["SimpleDynamicsModel","AutoregressiveModel" ,"ProbsDeltaDynamicsModel", "AutoregressiveDeltaModel"]
    """
    ensembles = ["EnsembleSDModel",  "EnsembleARModel", "EnsemblePDDModel", "EnsembleARDModel"]
    all_stats_models = {model: all_stats[model] for model in models}
    all_stats_ensembles = {model: all_stats[model] for model in ensembles}
    
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "spearman_corr", title=f" {reward[7:]}-reward \n Spearman Correlation", 
                          save_path=f"plots/spearman_corr_comparison_{reward}.pdf")
    # abs
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "abs", title=f" {reward[7:]}-reward \n Absolute Error", 
                          save_path=f"plots/abs_comparison_{reward}.pdf")
    # regret@1
    plot_stats_comparison(all_stats_models
                          , all_stats_ensembles, 
                          "regret@1", title=f" {reward[7:]}-reward \n Regret@1", 
                          save_path=f"plots/regret@1_comparison_{reward}.pdf")
    """
    # print files not found with warning if > 0
    if files_not_founds > 0:
        print(f"Warning: {files_not_founds} files not found")
    else:
        print(f"All files found for {reward}")