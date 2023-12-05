module load scipy-stack/2023b
pip install --no-index minigrid gymnasium hydra_core wandb submitit seaborn torch==1.9.1
pip install --no-index hydra-submitit-launcher --upgrade
pip install --no-index src/gym-monitor/.