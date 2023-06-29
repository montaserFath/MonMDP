# Monitored Markov Decision Process

`python main.py` for single run

- `-m` for sequential sweep defined in config file
- `-m hydra/launcher=joblib` for parallel sweep
- `wandb.project=cool_name wandb.mode=offline` to define WandB args
- `environment.id=FrozenLake` to manually define environment
- `monitor.id=BinaryMonitor` to manually define the monitor

Full example
```
python main -m hydra/launcher=joblib wandb.mode=offline wandb.project=test environment.id=FrozenLake monitor.id=BinaryMonitor
```
