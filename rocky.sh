#!/bin/bash
# Single-seed CPU run on the rocky cluster (binary monitor).
#   sbatch rocky.sh                  # seed 0, non-stationary reward
#   sbatch rocky.sh 3                # seed 3, non-stationary reward
#   sbatch rocky.sh 3 stationary     # seed 3, stationary baseline (original env reward)
#   sbatch rocky.sh 3 nonstationary 10000 true    # seed 3, reward changes every 10k steps, monitor cost varies too
# Args: SEED [REWARD=nonstationary] [PERIOD=from reward_change.yaml] [VARYMON=false]
# Results: general_models/.../nonstationary/period_<P>_cycle_varymon_<True|False>/ (stationary: the folder above it)
#   sbatch --time=12:00:00 -p gpubase_bynode_b2 rocky.sh   # longer run: pick a partition whose limit fits
# Partitions: b1 3h, b2 12h, b3 1d, b4 3d, b5 7d. For all seeds at once use the Hydra launcher instead:
#   python run_plants_env.py -m hydra/launcher=slurm_rocky ...
#SBATCH --job-name=monmdp
#SBATCH --account=aip-gburlet
#SBATCH --time=0-02:59
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --output=/scratch/montaser/MonMDP/slurm_logs/%j.out
#   renamed below to slurm_logs/<jobid>_<YYYY-MM-DD_HH-MM-SS>_<tag>_seed<seed>.out (the folder must exist before sbatch)

SEED=${1:-0}
REWARD=${2:-nonstationary}   # nonstationary | stationary
PERIOD=${3:-}                # nonstationary only: env steps between reward changes (default: reward_change.yaml)
VARYMON=${4:-false}          # nonstationary only: true = monitor cost follows the tables, false = fixed

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED' (usage: rocky.sh SEED [REWARD] [PERIOD] [VARYMON])"; exit 1; }
case "$REWARD" in
  nonstationary)
    [[ "$VARYMON" == true || "$VARYMON" == false ]] || { echo "VARYMON must be true or false, got '$VARYMON'"; exit 1; }
    REWARD_ARGS="+nonstationary=reward_change nonstationary.vary_monitor_cost=$VARYMON"
    if [ -n "$PERIOD" ]; then
      [[ "$PERIOD" =~ ^[0-9]+$ ]] || { echo "PERIOD must be an integer, got '$PERIOD'"; exit 1; }
      REWARD_ARGS="$REWARD_ARGS nonstationary.period_timesteps=$PERIOD"
    fi
    TAG="nonstationary_p${PERIOD:-yaml}_varymon${VARYMON}" ;;
  stationary)
    [ -z "$PERIOD" ] || { echo "PERIOD/VARYMON only apply to nonstationary runs"; exit 1; }
    REWARD_ARGS=""
    TAG="stationary" ;;
  *) echo "unknown reward mode '$REWARD' (nonstationary | stationary)"; exit 1 ;;
esac

# Slurm keeps writing to the same open file after a rename, including its own TIMEOUT / CANCELLED messages
LOG_DIR=/scratch/montaser/MonMDP/slurm_logs
if [ -n "$SLURM_JOB_ID" ] && [ -f "$LOG_DIR/$SLURM_JOB_ID.out" ]; then
  mv "$LOG_DIR/$SLURM_JOB_ID.out" "$LOG_DIR/${SLURM_JOB_ID}_$(date +%Y-%m-%d_%H-%M-%S)_${TAG}_seed${SEED}.out"
fi

source /scratch/montaser/miniforge3/bin/activate mmdp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /scratch/montaser/MonMDP

python run_plants_env.py $REWARD_ARGS \
  experiment.rng_seed="$SEED" experiment.training_timesteps=1e6 \
  agent.critic.device=cpu agent.critic.reward_model.device=cpu \
  +agent.critic.reward_model.flatten=True
