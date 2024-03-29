import os
import json

def aggregate_results(directory):
    # Traverse the user-provided directory to find all seed subfolders
    for seed_folder in os.listdir(directory):
        seed_path = os.path.join(directory, seed_folder)
        if os.path.isdir(seed_path):
            # Initialize the aggregated results dictionary
            aggregated_results = {}
            
            # For each seed subfolder, look for agent subfolders
            for agent_folder in os.listdir(seed_path):
                agent_path = os.path.join(seed_path, agent_folder)
                if os.path.isdir(agent_path):
                    results_path = os.path.join(agent_path, "results")
                    if os.path.exists(results_path) and os.path.isdir(results_path):
                        # Assuming there's only one .json file in the results folder
                        for file in os.listdir(results_path):
                            if file.endswith(".json"):
                                with open(os.path.join(results_path, file), 'r') as f:
                                    # Load the dictionary from the JSON file
                                    agent_results = json.load(f)
                                    # Save the dictionary, keyed by the agent name
                                    aggregated_results[agent_folder] = agent_results[agent_folder]
            
            # After processing all agents for the seed, save the aggregated dictionary
            if aggregated_results:
                with open(os.path.join(seed_path, "reward_progress.json"), 'w') as f:
                    json.dump(aggregated_results, f, indent=4)

# Replace "your_directory_path_here" with the path to the directory containing the seed subfolders
directory_path = "/home/fabian/msc/f110_dope/ws_release/experiments/runs_fqe_4_0.0001_0.005_0.0/QFitterDD/f110-real-stoch-v2/250/on-policy"
aggregate_results(directory_path)

print("Aggregation complete.")