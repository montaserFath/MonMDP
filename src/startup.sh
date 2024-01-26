module load scipy-stack/2023b
pip install minigrid gymnasium hydra_core wandb submitit seaborn torch scipy pygame==2.5.0
pip install hydra-submitit-launcher --upgrade
pip install --no-index src/gym-monitor/.
screen
# maybe uninstall then install numpy