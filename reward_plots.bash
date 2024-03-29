#!/bin/bash
for reward_file in "reward_lifetime.json" "reward_min_act.json" "reward_progress.json"
do
  python compute_reward_from_ds.py --target_reward="$reward_file"
done