# Monitored Markov Decision Process

`python main.py` for single run

- `-m` for sequential sweep defined in config file
- `-m hydra/launcher=joblib` for parallel sweep with Joblib
- `-m hydra/launcher=submitit_slurm` for parallel sweep with SLURM
- `wandb.project=cool_name wandb.mode=offline` to define WandB args
- `environment.id=FrozenLake` to manually define environment
- `monitor.id=BinaryMonitor` to manually define the monitor

Full example
```
python main -m hydra/launcher=joblib wandb.mode=offline wandb.project=test environment.id=FrozenLake monitor.id=BinaryMonitor
```

To install the our environments:
- `pip install -e envs`
- Then `gymnasium.make('monitor/ToyWorld-v0')` or `gymnasium.make('monitor/MonitorGrid-v0')`
