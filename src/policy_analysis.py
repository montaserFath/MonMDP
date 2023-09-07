import numpy as np
import matplotlib
import matplotlib.pylab as plt

ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}


def policy_performance(log_dir: str, cell_size: tuple = (3, 3), scale: float = 0.25, save_fig: bool = False) -> None:
    """
    plot policy actions as arrows in grid environment
    """
    traj = np.load(log_dir + "/trajectories.npy", allow_pickle=True)[()]
    states, actions = traj[0]["states"], traj[0]["actions"]  # TODO(Monta): remove hardcoded 0

    plt.figure(figsize=cell_size)
    plt.hlines(np.arange(cell_size[1] + 1) - scale * 2, -scale * 2, cell_size[0] - scale * 2, color="black")
    plt.vlines(np.arange(cell_size[0] + 1) - scale * 2, -scale * 2, cell_size[1] - scale * 2, color="black")

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
    plt.legend(handles=[mon_off, mon_on], fontsize=8)
    plt.xlim(-scale * 2, cell_size[0] - scale * 2)
    plt.ylim(-scale * 2, cell_size[1] - scale * 2)
    plt.axis("off")
    plt.tight_layout()
    if save_fig:
        plt.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)
