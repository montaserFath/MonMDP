"""Plot training curves for hypper-parameters sweep"""
import numpy as np
import matplotlib.pylab as plt
from policy_analysis import plot_policy_reward, set_blind_colors

COLORS, _ = set_blind_colors()


def plot_joint_reward_sweep(
    env_id: str,
    epsilon: float,
    q_lr: float,
    reward_lrs: list,
    n_seed: int = 30,
    legend: bool = False,
    save_fig: bool = False,
):
    """Plot joint reward for a policy with different parameters and seeds"""
    if env_id not in ["Simple", "Penalty", "Button"]:
        raise ValueError("The environment is not implemented yet")
    fig = plt.figure(figsize=(7, 4))
    # Load saved joint reward arrays
    rewards = []
    for count, r_lr in enumerate(reward_lrs):
        for seed in range(n_seed):
            rewards.append(
                np.load(
                    "models/3_3/{}/reward_model/eps_{}/q_lr_{}/reward_lr_{}/training_joint_reward_{}.npy".format(
                        env_id, epsilon, q_lr, r_lr, seed), allow_pickle=True)[()]
            )
        label = r"$R_\alpha = {}$".format(r_lr)
        fig = plot_policy_reward(fig, rewards, label=label, color=COLORS[count + 3], alpha=1)
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(1, 4))
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-20, 1.0)  # max reward is 1, min reward is -20
    if legend:
        plt.legend(fontsize=11, bbox_to_anchor=(1.1, 1.25), ncol=3, fancybox=True)
    plt.title(r"$\epsilon = {}$, $Q_\alpha = {}$".format(epsilon, q_lr), fontsize=12)
    plt.tight_layout()
    if save_fig:
        fig.savefig(
            "models/3_3/{}/reward_model/eps_{}/q_lr_{}/training_joint_reward.pdf".format(env_id, epsilon, q_lr),
            dpi=300,
        )


if __name__ == "__main__":
    for eps in [0.05, 0.1, 0.2]:
        for q_lr in [1.0]:
            legend = True if eps == 0.2 and q_lr == 0.1 else False
            plot_joint_reward_sweep("Penalty", eps, q_lr, [1.0], legend=legend, save_fig=True)
