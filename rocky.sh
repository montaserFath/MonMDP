#!/bin/bash
# Single-seed CPU run on the rocky cluster (binary monitor + non-stationary reward).
#   sbatch rocky.sh                  # seed 0
#   sbatch rocky.sh 3                # seed 3
#   sbatch --time=12:00:00 -p gpubase_bynode_b2 rocky.sh   # longer run: pick a partition whose limit fits
# Partitions: b1 3h, b2 12h, b3 1d, b4 3d, b5 7d. For all seeds at once use the Hydra launcher instead:
#   python run_plants_env.py -m hydra/launcher=slurm_rocky ...
#SBATCH --job-name=monmdp
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=slurm-%j.out

SEED=${1:-0}

source /scratch/montaser/miniforge3/bin/activate mmdp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /scratch/montaser/MonMDP

python run_plants_env.py +nonstationary=reward_change \
  experiment.rng_seed="$SEED" experiment.training_timesteps=1e6 \
  agent.critic.device=cpu agent.critic.reward_model.device=cpu \
  +agent.critic.reward_model.flatten=True
