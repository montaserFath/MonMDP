import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns


ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}
STATES = np.arange(9)
JOINT_STATES = [
    "(0,0)", "(1,0)", "(2,0)", "(3,0)", "(4,0)", "(5,0)", "(6,0)", "(7,0)", "(8,0)",
    "(0,1)", "(1,1)", "(2,1)", "(3,1)", "(4,1)", "(5,1)", "(6,1)", "(7,1)", "(8,1)",
]
# JOINT_STATES = [
#     "(0,off)", "(1,off)", "(2,off)", "(3,off)", "(4,off)", "(5,off)", "(6,off)", "(7,off)", "(8,off)",
#     "(0,on)", "(1,on)", "(2,on)", "(3,on)", "(4,on)", "(5,on)", "(6,on)", "(7,on)", "(8,on)",
# ]
MDP_ACTIONS = [r"$\leftarrow$", r"$\downarrow$", r"$\rightarrow$", r"$\uparrow$"]
JOINT_ACTIONS = [
    r"($\leftarrow$,0)",
    r"($\downarrow$,0)",
    r"($\rightarrow$,0)",
    r"($\uparrow$,0)",
    r"($\leftarrow$,1)",
    r"($\downarrow$,1)",
    r"($\rightarrow$,1)",
    r"($\uparrow$,1)",
]
COLORS = ["r", "g", "b", "y", "black"]
ALPHAS = [0.6, 0.8, 0.6, 0.9, 0.5]
N_TRAIN_EP = 10000
BASELINES = {
    "reward_model": "Reward Model",
    "q_monitor_joint": r"$Q_{joint}$",
    "q_monitor_sequential": r"$Q_{Sequential}$",
    "q_mdp": r"$Q_{ignore}$",
    "zero_reward": r"$Q_{\bot=0}$",
}


def plot_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Q-Table values as a heatmap"""
    q_table = np.round(np.load(log_dir + "/critic_q_table.npy"), 2)
    y_sticks = JOINT_STATES if q_table.shape[0] == 18 else np.arange(q_table.shape[0])

    fig = plt.figure(figsize=(7, 8 if len(y_sticks) == 18 else 5))
    ax = sns.heatmap(q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax.set_xlabel("Actions", fontsize=15)
    ax.set_ylabel("States", fontsize=15)
    ax.set_xticklabels(MDP_ACTIONS if q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    ax.set_yticks(0.5 + np.arange(len(y_sticks)))
    ax.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 18 else 13)
    plt.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/q_table_heatmap.pdf", dpi=300)


def plot_mdp_mon_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Q-Table values as a heatmap"""
    mdp_q_table = np.round(np.load(log_dir + "/mdp_q_table.npy"), 2)
    mon_q_table = np.round(np.load(log_dir + "/monitor_q_table.npy"), 2)
    y_sticks = JOINT_STATES if mon_q_table.shape[0] == 18 else np.arange(mon_q_table.shape[0])

    fig = plt.figure(figsize=(4, 5))
    ax = sns.heatmap(mdp_q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax.set_xlabel("Actions", fontsize=15)
    ax.set_ylabel("States", fontsize=15)
    ax.set_xticklabels(MDP_ACTIONS, fontsize=13)
    ax.set_yticks(0.5 + np.arange(len(STATES)))
    ax.set_yticklabels(STATES, fontsize=13)
    fig.tight_layout()

    fig_1 = plt.figure(figsize=(7, 8 if len(y_sticks) == 18 else 5))
    ax_1 = sns.heatmap(mon_q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax_1.set_xlabel("Actions", fontsize=15)
    ax_1.set_ylabel("States", fontsize=15)
    ax_1.set_xticklabels(MDP_ACTIONS if mon_q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    ax_1.set_yticks(0.5 + np.arange(len(y_sticks)))
    ax_1.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 18 else 13)
    fig_1.tight_layout()

    if save_fig:
        fig.savefig(log_dir + "/mdp_q_table_heatmap.pdf", dpi=300)
        fig_1.savefig(log_dir + "/monitor_q_table_heatmap.pdf", dpi=300)


def plot_reward_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Predictive reward table values as a heatmap"""
    fig = plt.figure(figsize=(4, 5))
    r_table = np.round(np.load(log_dir + "/reward_model_table.npy"), 2)
    ax_r = sns.heatmap(r_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax_r.set_xlabel("Actions", fontsize=15)
    ax_r.set_ylabel("States", fontsize=15)
    ax_r.set_xticklabels(MDP_ACTIONS, fontsize=15)
    ax_r.set_yticklabels(np.arange(r_table.shape[0]), fontsize=15)
    fig.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/predictive_reward_table_heatmap.pdf", dpi=300)


def plot_policy_actions(
        log_dir: str,
        env_name: str,
        cell_size: tuple = (3, 3),
        scale: float = 0.25,
        traj_n: int = 0,
        save_fig: bool = False,
) -> None:
    """
    plot policy actions as arrows in grid environment
    """
    if env_name not in ["TreasureHunt-Simple-v0", "TreasureHunt-Fire-v0", "TreasureHunt-Switch-v0"]:
        raise NotImplemented

    monitor_on_ind, monitor_off_ind = None, None
    traj = np.load(log_dir + "/trajectories.npy", allow_pickle=True)[()]
    if env_name == "TreasureHunt-Switch-v0":
        while monitor_on_ind is None:
            for i in range(20):
                if traj[i]["states"][0, 1] == 1:
                    monitor_on_ind = i
        while monitor_off_ind is None:
            for i in range(20):
                if traj[i]["states"][0, 1] == 0:
                    monitor_off_ind = i
        on_states, on_actions = traj[monitor_on_ind]["states"], traj[monitor_on_ind]["actions"]
        off_states, off_actions = traj[monitor_off_ind]["states"], traj[monitor_off_ind]["actions"]
        on_fig = plot_env_actions(env_name, on_states, on_actions, cell_size, scale)
        off_fig = plot_env_actions(env_name, off_states, off_actions, cell_size, scale)
        if save_fig:
            on_fig.savefig(log_dir + "/final_policy_performance_on.pdf", dpi=300)
            off_fig.savefig(log_dir + "/final_policy_performance_off.pdf", dpi=300)
    else:
        states, actions = traj[traj_n]["states"], traj[traj_n]["actions"]
        fig = plot_env_actions(env_name, states, actions, cell_size, scale)
        # fig.tight_layout()
        if save_fig:
            fig.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)


def plot_env_actions(env_name: str, states, actions, cell_size: tuple = (3, 3), scale: float = 0.25):
    # load images
    fire_img = plt.imread("img/fire_img.png")
    agent_img = plt.imread("img/agent_img.png")
    gold_img = plt.imread("img/gold_img.png")
    switch_img = plt.imread("img/switch_img.png")
    shift = scale * 2

    fig = plt.figure(figsize=cell_size)
    plt.hlines(np.arange(cell_size[1] + 1) - shift, -shift, cell_size[0] - shift, color="black")
    plt.vlines(np.arange(cell_size[0] + 1) - shift, -shift, cell_size[1] - shift, color="black")

    for i, action in enumerate(actions):
        pos = np.array([states[i, 0] // cell_size[0], states[i, 0] % cell_size[0]])  # MDP state
        pos[0] = np.abs(pos[0] - cell_size[0] + 1)
        line_c = "r" if action[1] == 0 else "b"  # Monitor action
        arrow = ARROWS[action[0]]
        plt.arrow(
            pos[1], pos[0], scale * arrow[0], scale * arrow[1], lw=1.8, head_length=0.1, head_width=0.15, color=line_c
        )

    mon_off = matplotlib.patches.Patch(color="r", label="Monitor Off")
    mon_on = matplotlib.patches.Patch(color="b", label="Monitor On")
    plt.legend(handles=[mon_off, mon_on], fontsize=7, loc=(0.28, 1.0))
    plt.xlim(-shift, cell_size[0] - shift)
    plt.ylim(-shift, cell_size[1] - shift)
    plt.axis("off")
    if env_name in ["TreasureHunt-Fire-v0", "TreasureHunt-Switch-v0"]:
        fire_1 = fig.add_axes([0.41, 0.67, 0.2, 0.2], anchor='NE', zorder=-1)
        fire_1.imshow(fire_img)
        fire_1.axis('off')

        fire_2 = fig.add_axes([0.41, 0.41, 0.2, 0.2], anchor='NE', zorder=-1)
        fire_2.imshow(fire_img)
        fire_2.axis('off')

    if env_name == "TreasureHunt-Switch-v0":
        switch = fig.add_axes([0.12, 0.1, 0.15, 0.15], anchor='NE', zorder=-1)
        switch.imshow(switch_img)
        switch.axis('off')

    agent = fig.add_axes([0.14, 0.75, 0.12, 0.12], anchor="NE", zorder=-1)
    agent.imshow(agent_img)
    agent.axis("off")

    gold = fig.add_axes([0.7, 0.7, 0.15, 0.15], anchor="NE", zorder=-1)
    gold.imshow(gold_img)
    gold.axis("off")

    return fig


def smooth_vector(vec: np.ndarray, factor=20) -> np.ndarray:
    box = np.ones(factor) / factor
    vec_smooth = np.convolve(vec, box, mode='same')
    return vec_smooth


def mean_time_period(vec: np.ndarray, period: int) -> np.ndarray:
    mean_vec = np.zeros(len(vec) // period)
    for i in range(len(vec) // period):
        mean_vec[i] = np.mean(vec[i * period: i * period + period])
    return mean_vec


def dict_to_numpy(reward: dict, gamma: float = 0.99) -> np.ndarray:
    discount = [gamma**i for i in range(100)]
    array = np.zeros(len(reward))
    for i, key in enumerate(reward.keys()):
        n = len(reward[key])
        array[i] = np.sum(np.array(reward[key]) * np.array(discount[:n]))
    return array


def plot_joint_reward(
        baselines: list,
        env_name: str,
        testing_freq: int = 10,
        save_fig: bool = False,
) -> None:
    train_freq = 100
    joint_rewards = np.zeros((len(baselines), N_TRAIN_EP))
    eval_joint_rewards = np.zeros((len(baselines), N_TRAIN_EP // testing_freq))
    x_axis = testing_freq * np.arange(N_TRAIN_EP // testing_freq)
    for i, base in enumerate(baselines):
            joint_rewards[i] = dict_to_numpy(np.load(
                "models/{}/{}/training_joint_reward.npy".format(env_name, base), allow_pickle=True
            )[()])
            eval_joint_rewards[i] = np.mean(
                np.load("models/{}/{}/evaluation_joint_reward.npy".format(env_name, base)), 1
            )
            eval_joint_rewards[i, -1] = eval_joint_rewards[i, -2]

    fig = plt.figure(figsize=(7, 4))
    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)
    for i in range(len(joint_rewards)):
        # plt.plot(smooth_vector(joint_rewards[i], 100), lw=2, alpha=alphas[i], color=colors[i], label=BASELINES[baselines[i]])
        mean_reward = mean_time_period(joint_rewards[i], train_freq)
        plt.plot(train_x_axis, mean_reward, lw=2, alpha=ALPHAS[i], color=COLORS[i], label=BASELINES[baselines[i]])
        plt.plot(train_x_axis, mean_time_period(joint_rewards[i], train_freq), "*", alpha=ALPHAS[i], color=COLORS[i])
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Training Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()

    eval_fig = plt.figure(figsize=(7, 4))

    for i in range(len(joint_rewards)):
        plt.plot(x_axis, eval_joint_rewards[i], lw=2, alpha=ALPHAS[i], color=COLORS[i], label=BASELINES[baselines[i]])
        # plt.scatter(x_axis, eval_joint_rewards[i], marker="*", s=5, alpha=alphas[i], color=colors[i])
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Testing Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig.savefig("models/{}/training_joint_reward.pdf".format(env_name), dpi=300)
        eval_fig.savefig("models/{}/evaluation_joint_reward.pdf".format(env_name), dpi=300)


def plot_joint_reward_seeds(env_name: str, baseline: str, save_fig: bool = False):
    n_seeds = 30
    train_freq = 100
    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)
    joint_reward = np.zeros((n_seeds, N_TRAIN_EP // train_freq))

    fig = plt.figure(figsize=(7, 4))
    for seed in range(n_seeds):
        reward = np.load(
            "models/{}/{}/training_joint_reward_{}.npy".format(env_name, baseline, seed), allow_pickle=True,
        )[()]
        joint_reward[seed] = mean_time_period(dict_to_numpy(reward) if isinstance(reward, dict) else reward, train_freq)
        plt.plot(train_x_axis, joint_reward[seed], lw=0.1)
        plt.scatter(train_x_axis, joint_reward[seed], marker="*", s=1)
    # mean, std = np.mean(joint_reward, 0), np.std(joint_reward, 0)
    # plt.plot(train_x_axis, mean, lw=0.5, c="r", alpha=0.9)
    # plt.plot(train_x_axis, mean, marker="*", c="r", alpha=0.9)
    # plt.fill_between(train_x_axis, mean + std, mean - std, facecolor="r", alpha=0.3)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_fig:
        fig.savefig("models/{}/{}/seeds_training_joint_reward.pdf".format(env_name, baseline), dpi=300)


def plot_joint_reward_seeds_baselines(env_name: str, baselines: list, save_fig: bool = False):
    n_seeds = 30
    train_freq = 100
    # joint_reward = np.zeros((len(baselines), n_seeds, n_train_steps // train_freq))
    mean = np.zeros((len(baselines), N_TRAIN_EP // train_freq))
    std = np.zeros((len(baselines), N_TRAIN_EP // train_freq))

    for i, baseline in enumerate(baselines):
        base_joint_reward = np.zeros((n_seeds, N_TRAIN_EP // train_freq))
        for seed in range(n_seeds):
            reward = np.load(
                "models/{}/{}/training_joint_reward_{}.npy".format(env_name, baseline, seed), allow_pickle=True,
            )[()]
            base_joint_reward[seed] = mean_time_period(
                dict_to_numpy(reward) if isinstance(reward, dict) else reward, train_freq,
            )
        mean[i], std[i] = np.mean(base_joint_reward, 0), np.std(base_joint_reward, 0)

    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)

    fig = plt.figure(figsize=(7, 4))
    for i in range(len(baselines)):
        plt.plot(train_x_axis, mean[i], lw=2, c=COLORS[i], alpha=ALPHAS[i], label=BASELINES[baselines[i]])
        plt.scatter(train_x_axis, mean[i], marker="*", c=COLORS[i], s=1, alpha=ALPHAS[i])
        plt.fill_between(train_x_axis, mean[i] + std[i], mean[i] - std[i], facecolor=COLORS[i], alpha=ALPHAS[i])
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig.savefig("models/{}/seeds_training_joint_reward.pdf".format(env_name), dpi=300)
