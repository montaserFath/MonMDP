"""Plot training curves for hypper-parameters sweep"""
import numpy as np
import matplotlib.pylab as plt
from policy_analysis import plot_policy_reward, set_blind_colors, plot_q_table_heatmap

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
    linestyles = ["solid", "", ""]
    markers = [None, "o", "v"]
    if env_id not in ["Simple", "Penalty", "Button"]:
        raise ValueError("The environment is not implemented yet")
    fig = plt.figure(figsize=(7, 4))
    count_all = 0
    for q_count, q_lr in enumerate(q_lrs):
        # Load saved joint reward arrays
        rewards = []
        for count, r_lr in enumerate(reward_lrs):
            for seed in range(n_seed):
                rewards.append(
                    np.load(
                        "models/9_9/{}/reward_model/env_{}/eps_{}/q_lr_{}/reward_lr_{}/training_joint_reward_{}.npy".format(
                            env_id, env_random, epsilon, q_lr, r_lr, seed
                        ),
                        allow_pickle=True,
                    )[()]
                )
            label = r" $Q_\alpha = {}$, $R_\alpha = {}$".format(q_lr, r_lr)
            alpha = 0.6 + q_count * 0.2
            fig = plot_policy_reward(
                fig, rewards, label=label, color=COLORS[count], alpha=1, lw=1, marker=markers[q_count], linestyle=linestyles[q_count]
            )
        label = r"$R_\alpha = {}$".format(r_lr)
        fig = plot_policy_reward(fig, rewards, label=label, color=COLORS[count + 3], alpha=1)
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(1, 4))
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-50, 2.0)  # max reward is 1, min reward is -20
    if legend:
        legend = plt.legend(fontsize=9, bbox_to_anchor=(0.9, -0.22), ncol=3, fancybox=True)
        fig_legend = legend.figure
        fig_legend.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
        fig_legend.savefig("models/9_9/{}/reward_model/env_{}/eps_{}/legend.pdf".format(env_id, env_random, epsilon), dpi=300, bbox_inches=bbox)
        # plt.legend(ncol=2)
    plt.title(r"env = {}, $\epsilon = {}$".format(env_random, epsilon), fontsize=12)
    plt.tight_layout()
    if save_fig:
        fig.savefig(
            "models/3_3/{}/reward_model/eps_{}/q_lr_{}/training_joint_reward.pdf".format(env_id, epsilon, q_lr),
            dpi=300,
        )


if __name__ == "__main__":
    env_random = 0.2
    for eps in ["dec"]:
        # for q_lr in [1.0, 0.5, 0.1]:
        #     legend = True if eps == 0.2 and q_lr == 0.1 else False
        plot_joint_reward_sweep(
            "Penalty", eps, [1.0, 0.5, 0.1], [1.0, 0.5, 0.1], env_random=env_random, legend=True, save_fig=True
        )
    # seed = 0
    # log_dir = "models/9_9/Penalty/reward_model/env_{}/eps_{}/q_lr_{}/reward_lr_{}".format(env_random, 0.05, 1.0, 1.0)
    # plot_q_table_heatmap(log_dir, seed=seed, grid_size=(9, 9), save_fig=True)