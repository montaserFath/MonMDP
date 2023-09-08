import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns


ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}
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


def plot_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Q-Table values as a heatmap"""
    q_table = np.load(log_dir + "/critic_q_table.npy")
    ax = sns.heatmap(q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax.set_xlabel("Actions", fontsize=15)
    ax.set_ylabel("States", fontsize=15)
    ax.set_xticklabels(MDP_ACTIONS if q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    ax.set_yticklabels(np.arange(q_table.shape[0]), fontsize=13)
    plt.tight_layout()
    if save_fig:
        plt.savefig(log_dir + "/q_table_heatmap.pdf", dpi=300)


def plot_reward_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Predictive reward table values as a heatmap"""
    fig = plt.figure(figsize=(4, 5))
    r_table = np.load(log_dir + "/reward_model_table.npy")
    ax_r = sns.heatmap(r_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax_r.set_xlabel("Actions", fontsize=15)
    ax_r.set_ylabel("States", fontsize=15)
    ax_r.set_xticklabels(MDP_ACTIONS, fontsize=15)
    ax_r.set_yticklabels(np.arange(r_table.shape[0]), fontsize=15)
    fig.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/predictive_reward_table_heatmap.pdf", dpi=300)


def plot_policy_actions(
        log_dir: str, env_name: str, cell_size: tuple = (3, 3), scale: float = 0.25, save_fig: bool = False
) -> None:
    """
    plot policy actions as arrows in grid environment
    """
    if env_name not in ["TreasureHunt-Simple-v0", "TreasureHunt-Fire-v0", "TreasureHunt-Switch-v0"]:
        raise NotImplemented
    # load images
    fire_img = plt.imread("img/fire_img.png")
    agent_img = plt.imread("img/agent_img.png")
    gold_img = plt.imread("img/gold_img.png")
    shift = scale * 2

    traj = np.load(log_dir + "/trajectories.npy", allow_pickle=True)[()]
    states, actions = traj[0]["states"], traj[0]["actions"]  # TODO(Monta): remove hardcoded 0

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
    if env_name == "TreasureHunt-Fire-v0":
        fire_1 = fig.add_axes([0.41, 0.67, 0.2, 0.2], anchor='NE', zorder=1)
        fire_1.imshow(fire_img)
        fire_1.axis('off')

        fire_2 = fig.add_axes([0.41, 0.41, 0.2, 0.2], anchor='NE', zorder=1)
        fire_2.imshow(fire_img)
        fire_2.axis('off')

    if env_name == "TreasureHunt-Switch-v0":
        raise NotImplemented

    agent = fig.add_axes([0.14, 0.75, 0.12, 0.12], anchor="NE", zorder=1)
    agent.imshow(agent_img)
    agent.axis("off")
    
    gold = fig.add_axes([0.7, 0.7, 0.15, 0.15], anchor="NE", zorder=1)
    gold.imshow(gold_img)
    gold.axis("off")

    # fig.tight_layout()
    if save_fig:
        plt.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)
