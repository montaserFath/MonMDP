module load scipy-stack/2023b
pip install --no-index minigrid gymnasium hydra_core wandb submitit seaborn torch scipy stable_baselines3
# pip install hydra-submitit-launcher --upgrade
pip install --no-index src/gym-monitor/.
screen
#python simple_env.py -m hydra/launcher=slurm_cc_graham