#!/bin/bash

# Initial seed value
seed=0

# Number of iterations to run the scripts
iterations=10

# Dynamics model variable
dynamics_model="AutoregressiveModel"

for ((i=0; i<iterations; i++))
do
    echo "Iteration $(($i + 1)) with seed $seed"

    # Call train_mb.py
    python train_mb.py --dynamics_model=$dynamics_model --split=off-policy --train --save_model --seed=$seed --update_steps=30_001 --skip_eval

    # Call reward_rollouts.py
    python reward_rollouts.py --dynamics_model=$dynamics_model --split=off-policy --model_checkpoint=model_${seed}_30000.pth --seed=$seed --save

    # Call create_plots.py
    python create_plots.py --dynamics_model=$dynamics_model --seed=$seed

    # Increment the seed for the next iteration
    seed=$(($seed + 1))
done