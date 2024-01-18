"""Evaluate a policy with different random seeds"""
import hydra
from src.actor import MonEpsilonGreedyOneAction, MonStateEpsilonGreedy
from src.critic import MonQTableOneAction, StateMonTable
from src.experiment import MonExperiment
from simple_env import wrappe_env


@hydra.main(version_base=None, config_path="configs", config_name="penalty_env")
def evaluate_policy(policy_dir: str, seed: int, n_episodes: int = 30, save_logs: bool = False, cfg=None) -> None:
    env = wrappe_env("gym_monitor/TreasureHunt-Penalty-v1", train=False, monitor_wrapper=True, cfg=cfg)
    critic = MonQTableOneAction(
        cfg["environment"]["id"], env.observation_space, env.action_space, policy_dir, **cfg.agent.critic
    )
    actor = MonEpsilonGreedyOneAction(critic, train=False, **cfg.agent.actor)
    experiment = MonExperiment(env, actor, critic, log_dir=policy_dir, **cfg.experiment)
    critic.load(policy_dir, seed=cfg.experiment.rng_seed)
    received_reward, proxy_reward, _, _, ep_len, traj = experiment.test(render=False, save_results=True)
