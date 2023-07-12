# Monitored Markov Decision Process

`python main.py` for single run

- `-m` for sequential sweep defined in config file
- `-m hydra/launcher=joblib` for parallel sweep with Joblib
- `-m hydra/launcher=slurm_cc_beluga` for parallel sweep with SLURM on CC (Beluga)
- `-m hydra/launcher=slurm_local` for parallel sweep with SLURM on your local machine
- `wandb.project=cool_name wandb.mode=offline` to define WandB args
- `environment.id=FrozenLake` to manually define environment
- `monitor.id=BinaryMonitor` to manually define the monitor

Note that when we submit jobs the python code will hang and wait until all jobs are done.
To detach the run from the current session do
`nohup python main.py -m hydra/launcher=slurm_cc_beluga &`

Full example
```
python main -m hydra/launcher=joblib wandb.mode=offline wandb.project=test environment.id=FrozenLake monitor.id=BinaryMonitor
```

To install and use our environments:
```
cd src/gym-monitor
pip install .
```

`env = gymnasium.make('Gym-Monitor/TreasureHunt-Easy-v0')`


For the lava experiment: `environment.id=MiniGrid-LavaCrossingS9N1-v0`
