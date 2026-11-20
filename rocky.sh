#!/bin/bash
# CPU runs on the rocky cluster (binary monitor), several seeds in parallel inside ONE job.
#   sbatch rocky.sh 0-4 stationary                       # stationary baseline, default buffer (1e6) / train every 20 eps
#   sbatch rocky.sh 0-4 stationary 10000                 # stationary control with the 10k-period buffer/interval
#   sbatch rocky.sh 0-4 nonstationary 10000 false        # reward changes every 10k steps, monitor cost fixed
#   sbatch rocky.sh 0-4 nonstationary 10000 true         # ... monitor cost varies with the reward table
#   sbatch rocky.sh 0-4 nonstationary 10000 false default  # same, but the old buffer / interval (not scaled)
# Args: SEEDS [REWARD=nonstationary] [PERIOD] [VARYMON=false] [SCALE=proportional]
#   SEEDS   "0-4", "0,2,7" or "3"; each seed is its own python process (2 threads), all run at the same time
#   PERIOD  env steps between reward changes (nonstationary: default from reward_change.yaml). For stationary runs
#           it only sets the scaled buffer / interval, as a matched control.
#   SCALE   proportional: replay buffer = BUF_PHASES x PERIOD steps (default 1 phase, so old tables are evicted)
#                         and train every PERIOD / 100 / UPDATES_PER_PHASE episodes (default 10 rounds per phase),
#                         so every period gets the same number of updates and the same data age per reward table.
#           default:      replay buffer 1e6, train every 20 episodes (the earlier runs).
#   BUF_PHASES=2 UPDATES_PER_PHASE=5 sbatch rocky.sh ...   # change the proportionality constants
# Results: general_models/.../[nonstationary/period_<P>_cycle_varymon_<V>/][buf_<B>_every_<K>/]
# Logs:    slurm_logs/<jobid>_<date>_<time>_<tag>.out (job), and one <...>_seed<s>.out per seed
# Partitions: b1 3h, b2 12h, b3 1d, b4 3d, b5 7d, e.g.  sbatch --time=12:00:00 -p gpubase_bynode_b2 rocky.sh ...
#SBATCH --job-name=monmdp
#SBATCH --account=aip-gburlet
#SBATCH --time=0-01:59
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --output=/scratch/montaser/MonMDP/slurm_logs/%j.out
# memory: ~3.7 GB per seed with the 1e6 buffer (5 seeds ~19 GB), well under 1 GB per seed with a 1-phase buffer
# cpus: 2 per seed; keep cpus-per-task = 2 x number of seeds

SEEDS_ARG=${1:-0}
REWARD=${2:-nonstationary}   # nonstationary | stationary
PERIOD=${3:-}
VARYMON=${4:-false}
SCALE=${5:-proportional}     # proportional | default
BUF_PHASES=${BUF_PHASES:-1}
UPDATES_PER_PHASE=${UPDATES_PER_PHASE:-10}
EP_LEN=100
REPO=/scratch/montaser/MonMDP
die() { echo "rocky.sh: $*" >&2; exit 1; }

# seeds: "0-4" | "0,2,7" | "3"
if [[ "$SEEDS_ARG" =~ ^([0-9]+)-([0-9]+)$ ]]; then
  SEEDS=$(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
elif [[ "$SEEDS_ARG" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  SEEDS=${SEEDS_ARG//,/ }
else
  die "SEEDS must look like 0-4, 0,2,7 or 3; got '$SEEDS_ARG'"
fi

[ -z "$PERIOD" ] || [[ "$PERIOD" =~ ^[0-9]+$ ]] || die "PERIOD must be an integer, got '$PERIOD'"
[[ "$SCALE" == proportional || "$SCALE" == default ]] || die "SCALE must be proportional or default, got '$SCALE'"
case "$REWARD" in
  nonstationary)
    [[ "$VARYMON" == true || "$VARYMON" == false ]] || die "VARYMON must be true or false, got '$VARYMON'"
    [ -n "$PERIOD" ] || PERIOD=$(awk '/^period_timesteps:/ {print $2}' "$REPO/configs/nonstationary/reward_change.yaml")
    ARGS="+nonstationary=reward_change nonstationary.vary_monitor_cost=$VARYMON nonstationary.period_timesteps=$PERIOD"
    TAG="nonstationary_p${PERIOD}_varymon${VARYMON}" ;;
  stationary)
    ARGS=""
    TAG="stationary${PERIOD:+_matchp$PERIOD}" ;;
  *) die "unknown reward mode '$REWARD' (nonstationary | stationary)" ;;
esac

if [[ "$SCALE" == proportional && -n "$PERIOD" ]]; then
  BUFFER=$(( PERIOD * BUF_PHASES ))
  (( PERIOD % (EP_LEN * UPDATES_PER_PHASE) == 0 )) || die "PERIOD must be a multiple of $EP_LEN x UPDATES_PER_PHASE"
  TRAIN_EVERY=$(( PERIOD / EP_LEN / UPDATES_PER_PHASE ))
  ARGS="$ARGS experiment.replay_buffer_size=$BUFFER experiment.train_every_episodes=$TRAIN_EVERY"
  TAG="${TAG}_buf${BUFFER}_every${TRAIN_EVERY}"
else
  TAG="${TAG}_defaultbuf"
fi

# Slurm keeps writing to the same open file after a rename, including its own TIMEOUT / CANCELLED messages
LOG_DIR=$REPO/slurm_logs
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
PREFIX="$LOG_DIR/${SLURM_JOB_ID:-local}_${STAMP}_${TAG}"
if [ -n "$SLURM_JOB_ID" ] && [ -f "$LOG_DIR/$SLURM_JOB_ID.out" ]; then
  mv "$LOG_DIR/$SLURM_JOB_ID.out" "${PREFIX}.out"
fi

source /scratch/montaser/miniforge3/bin/activate mmdp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO"

echo "$(date) seeds: $(echo $SEEDS) | overrides: $ARGS"
declare -A PIDS
for SEED in $SEEDS; do
  python run_plants_env.py $ARGS \
    experiment.rng_seed="$SEED" experiment.training_timesteps=1e6 \
    agent.critic.device=cpu agent.critic.reward_model.device=cpu \
    +agent.critic.reward_model.flatten=True \
    > "${PREFIX}_seed${SEED}.out" 2>&1 &
  PIDS[$SEED]=$!
  sleep 2  # distinct Hydra output dirs (timestamped to the second)
done

FAILED=0
for SEED in "${!PIDS[@]}"; do
  if wait "${PIDS[$SEED]}"; then echo "$(date) seed $SEED done"
  else echo "$(date) seed $SEED FAILED (exit $?), see ${PREFIX}_seed${SEED}.out"; FAILED=1; fi
done
exit $FAILED
