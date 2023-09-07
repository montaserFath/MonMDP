import numpy as np
import matplotlib
import matplotlib.pylab as plt

ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}


def policy_performance(log_dir: str, cell_size: tuple = (3, 3), scale: float = 0.25, save_fig: bool = False) -> None:
    """
    plot policy actions as arrows in grid environment
    """
    # load images
    fire_img = plt.imread("img/fire_img.png")
    agent_img = plt.imread("img/agent_img.png")
    gold_img = plt.imread("img/gold_img.png")

    traj = np.load(log_dir + "/trajectories.npy", allow_pickle=True)[()]
    states, actions = traj[0]["states"], traj[0]["actions"]  # TODO(Monta): remove hardcoded 0

    fig = plt.figure(figsize=cell_size)
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
    plt.legend(handles=[mon_off, mon_on], fontsize=7, loc=(0.28, 1.0))
    plt.xlim(-scale * 2, cell_size[0] - scale * 2)
    plt.ylim(-scale * 2, cell_size[1] - scale * 2)
    plt.axis("off")

    fire_1 = fig.add_axes([0.41, 0.67, 0.2, 0.2], anchor='NE', zorder=1)
    fire_1.imshow(fire_img)
    fire_1.axis('off')

    fire_2 = fig.add_axes([0.41, 0.41, 0.2, 0.2], anchor='NE', zorder=1)
    fire_2.imshow(fire_img)
    fire_2.axis('off')
    
    agent = fig.add_axes([0.14, 0.75, 0.12, 0.12], anchor="NE", zorder=1)
    agent.imshow(agent_img)
    agent.axis("off")
    
    gold = fig.add_axes([0.7, 0.7, 0.15, 0.15], anchor="NE", zorder=1)
    gold.imshow(gold_img)
    gold.axis("off")

    # fig.tight_layout()
    if save_fig:
        plt.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)
