"""Plotting functions for the training/testing, heatmaps for the q-tables, policy actions, and final trajectories"""
import colorsys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import scipy
import seaborn as sns


ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}
JOINT_STATES = [
    "(0,0)",
    "(1,0)",
    "(2,0)",
    "(3,0)",
    "(4,0)",
    "(5,0)",
    "(6,0)",
    "(7,0)",
    "(8,0)",
    "(0,1)",
    "(1,1)",
    "(2,1)",
    "(3,1)",
    "(4,1)",
    "(5,1)",
    "(6,1)",
    "(7,1)",
    "(8,1)",
]
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
N_TRAIN_TIMESTEPS = 250000
BASELINES = {
    "q_learning": r"$Q_{Oracle}$",
    "q_learning_mon": r"$Q_{Oracle}$",
    "reward_model": r"$Q_{Reward\,Model}$",
    "reward_model_neg": r"$Q_{Reward\,Model}, r_0 = -10$",
    "reward_model_0": r"$Q_{Reward\,Model}, r_0 = 0$",
    "reward_model_pos": r"$Q_{Reward\,Model}, r_0 = +1$",
    "q_monitor_joint": r"$Q_{Joint}$",
    "q_monitor_sequential": r"$Q_{Sequential}$",
    "q_mdp": r"$Q_{Ignore}$",
    "zero_reward": r"$Q_{\bot=0}$",
    "zero_reward_0": r"$Q_{\bot=0}$",
    "zero_reward_neg": r"$Q_{\bot=-10}$",
    "zero_reward_pos": r"$Q_{\bot=1}$",
}
CELL_SIZE = (10, 10)
STATES = np.arange(int(CELL_SIZE[0] * CELL_SIZE[1]))
SCALE = 0.25


def set_blind_colors() -> (list, list):
    """Get Blind colors friendly colors"""
    dark_hues = [0, 0.1, 0.4, 0.55, 0.65, 0.75, 0.9]
    dark_lightness = [0.4, 0.2, 0.15, 0.4, 0.4, 0.4, 0.4]
    light_hues = [1, 0.1, 0.3, 0.5, 0.6, 0.75, 0.85]
    light_lightness = [0.75, 0.6, 0.45, 0.6, 0.65, 0.8, 0.55]
    dark_colors = [colorsys.hls_to_rgb(h, dark_lightness[i], 1) for i, h in enumerate(dark_hues)]
    sns.palplot(dark_colors)
    light_colors = [colorsys.hls_to_rgb(h, light_lightness[i], 1) for i, h in enumerate(light_hues)]
    sns.palplot(light_colors)
    return light_colors, dark_colors


COLORS, _ = set_blind_colors()
ALPHAS = [1.0, 1.0, 0.4, 0.6, 1.0, 1.0]  # np.ones(len(COLORS))


def plot_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Monitor Q-Table for values as a heatmap"""
    q_table = np.round(np.load(log_dir + "/critic_q_table_1.npy"), 2)
    y_sticks = JOINT_STATES if q_table.shape[0] == 18 else np.arange(q_table.shape[0])

    fig = plt.figure(figsize=(7, 8 if len(y_sticks) == 18 else 5))
    axis = sns.heatmap(
        q_table,
        cmap="crest",
        annot=True,
        linewidth=0.1,
        fmt="g",
        annot_kws={"fontsize": 12},
    )
    axis.set_xlabel("Actions", fontsize=15)
    axis.set_ylabel("States", fontsize=15)
    axis.set_xticklabels(MDP_ACTIONS if q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    axis.set_yticks(0.5 + np.arange(len(y_sticks)))
    axis.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 18 else 13)
    plt.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/q_table_heatmap.pdf", dpi=300)


def plot_mdp_mon_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot MDP and Monitor Q-Tables values as a heatmap"""
    n_states = int(CELL_SIZE[0] * CELL_SIZE[1])
    n_actions = 4
    mdp_q_table = np.round(np.load(log_dir + "/mdp_q_table_1.npy"), 2)
    mon_q_table = np.round(np.load(log_dir + "/monitor_q_table_1.npy"), 2)
    y_sticks = JOINT_STATES if mon_q_table.shape[0] == 2 * n_states else np.arange(mon_q_table.shape[0])

    fig = plt.figure(figsize=(4, 5))
    axis = sns.heatmap(
        mdp_q_table,
        cmap="crest",
        annot=True,
        linewidth=0.1,
        fmt="g",
        annot_kws={"fontsize": 12},
    )
    axis.set_xlabel("Actions", fontsize=15)
    axis.set_ylabel("States", fontsize=15)
    axis.set_xticklabels(MDP_ACTIONS, fontsize=13)
    axis.set_yticks(0.5 + np.arange(len(STATES)))
    axis.set_yticklabels(STATES, fontsize=13)
    fig.tight_layout()

    fig_1 = plt.figure(figsize=(7, 8 if len(y_sticks) == 2 * n_states else 5))
    axis_1 = sns.heatmap(
        mon_q_table,
        cmap="crest",
        annot=True,
        linewidth=0.1,
        fmt="g",
        annot_kws={"fontsize": 12},
    )
    axis_1.set_xlabel("Actions", fontsize=15)
    axis_1.set_ylabel("States", fontsize=15)
    axis_1.set_xticklabels(MDP_ACTIONS if mon_q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    axis_1.set_yticks(0.5 + np.arange(len(y_sticks)))
    axis_1.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 2 * n_states else 13)
    fig_1.tight_layout()

    combined_q_table = np.zeros_like(mon_q_table)
    if len(y_sticks) == 2 * n_states:  # Button env & StateMonitor
        combined_q_table[:n_states] = mdp_q_table + mon_q_table[:n_states]
        combined_q_table[n_states:] = mdp_q_table + mon_q_table[n_states:]
    else:  # Simple/penalty & BinaryMonitor
        combined_q_table[:, :n_actions] = mdp_q_table + mon_q_table[:, :n_actions]
        combined_q_table[:, n_actions:] = mdp_q_table + mon_q_table[:, n_actions:]

    fig_2 = plt.figure(figsize=(7, 8 if len(y_sticks) == 2 * n_states else 5))
    axis_2 = sns.heatmap(
        combined_q_table,
        cmap="crest",
        annot=True,
        linewidth=0.1,
        fmt="g",
        annot_kws={"fontsize": 12},
    )
    axis_2.set_xlabel("Actions", fontsize=15)
    axis_2.set_ylabel("States", fontsize=15)
    axis_2.set_xticklabels(MDP_ACTIONS if mon_q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    axis_2.set_yticks(0.5 + np.arange(len(y_sticks)))
    axis_2.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 2 * n_states else 13)
    fig_2.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/mdp_q_table_heatmap.pdf", dpi=300)
        fig_1.savefig(log_dir + "/monitor_q_table_heatmap.pdf", dpi=300)
        fig_2.savefig(log_dir + "/combined_q_table_heatmap.pdf", dpi=300)


def plot_reward_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Predictive reward table values as a heatmap"""
    fig = plt.figure(figsize=(4, 5))
    r_table = np.round(np.load(log_dir + "/reward_model_table_1.npy"), 2)
    ax_r = sns.heatmap(
        r_table,
        cmap="crest",
        annot=True,
        linewidth=0.1,
        fmt="g",
        annot_kws={"fontsize": 12},
    )
    ax_r.set_xlabel("Actions", fontsize=15)
    ax_r.set_ylabel("States", fontsize=15)
    ax_r.set_xticklabels(MDP_ACTIONS, fontsize=15)
    # ax_r.set_yticklabels(np.arange(r_table.shape[0]), fontsize=15 if CELL_SIZE == (3, 3) else 5)
    fig.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/predictive_reward_table_heatmap.pdf", dpi=300)


# pylint: disable=too-many-locals
def plot_policy_trajectory(
        log_dir: str,
        env_name: str,
        traj_n: int = 1,
        save_fig: bool = False,
) -> None:
    """
    plot final actions actions as arrows in grid environment
    """
    if env_name not in [
        "TreasureHunt-Simple-v0", "TreasureHunt-Penalty-v0", "TreasureHunt-Penalty-v1", "TreasureHunt-Button-v0", "TreasureHunt-Button-v1",
    ]:
        raise NotImplemented

    monitor_on_ind, monitor_off_ind = None, None
    traj = np.load(log_dir + "/trajectories.npy", allow_pickle=True)[()]
    if env_name == "TreasureHunt-Button-v0":
        while monitor_on_ind is None:
            for i in range(len(traj.keys())):
                if traj[i]["states"][0, 1] == 1:
                    monitor_on_ind = i
        while monitor_off_ind is None:
            for i in range(len(traj.keys())):
                if traj[i]["states"][0, 1] == 0:
                    monitor_off_ind = i
        on_states, on_actions = traj[monitor_on_ind]["states"], traj[monitor_on_ind]["actions"]
        off_states, off_actions = traj[monitor_off_ind]["states"], traj[monitor_off_ind]["actions"]
        fig = plot_env_actions(
            env_name.split("-")[1],
            {"on": on_states, "off": off_states},
            {"on": on_actions, "off": off_actions},
            plot_both=True,
            boarder_color="green" if log_dir.split("/")[2] == "reward_model" else "r",
        )
    else:
        states, actions = traj[traj_n]["states"], traj[traj_n]["actions"]
        boarder_color = "r" if log_dir.split("/")[2] in ["q_mdp", "zero_reward", "zero_reward_0"] else "green"
        fig = plot_env_actions(env_name.split("-")[1], states, actions, boarder_color=boarder_color)
    if save_fig:
        fig.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)


# pylint: disable=too-many-locals
def plot_policy_button(baselines: list, save_fig: bool = False) -> None:
    """Plot policy actions for each state in the button environment"""
    n_states = int(CELL_SIZE[0] * CELL_SIZE[1])
    for base in baselines:
        boarder_color = "green" if base == "reward_model" else "r"
        mdp_q_table = None
        if base in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q_table = np.load("models/Button/{}/mdp_q_table_1.npy".format(base))
            q_table = np.load("models/Button/{}/monitor_q_table_1.npy".format(base))
        else:
            q_table = np.load("models/Button/{}/critic_q_table_1.npy".format(base))
        states = np.zeros((2 * n_states, 2))
        states[:n_states, 0], states[n_states:, 0] = np.arange(n_states), np.arange(n_states)
        states[n_states:, 1] = 1

        on_actions, off_actions = np.zeros(n_states), np.zeros(n_states)
        for j in range(n_states):
            if mdp_q_table is None:
                off_actions[j] = np.argmax(q_table[get_state_ind(states[j], n_states)])
                on_actions[j] = np.argmax(q_table[get_state_ind(states[j + n_states], n_states)])
            else:
                if base == "q_monitor_sequential":
                    off_actions[j] = np.argmax(mdp_q_table[int(states[j, 0])])
                    on_actions[j] = np.argmax(mdp_q_table[int(states[j, 0])])
                else:
                    mdp_value = mdp_q_table[int(states[j, 0])]
                    off_q_value = mdp_value + q_table[get_state_ind(states[j], n_states)]
                    on_q_value = mdp_value + q_table[get_state_ind(states[j + n_states], n_states)]
                    off_actions[j], on_actions[j] = np.argmax(off_q_value), np.argmax(on_q_value)

        on_actions = np.stack((on_actions, np.zeros(n_states)), 1)
        off_actions = np.stack((off_actions, np.zeros(n_states)), 1)
        fig = plot_env_actions(
            "Button",
            {"on": states[n_states:], "off": states[:n_states]},
            {"on": on_actions, "off": off_actions},
            plot_both=True,
            boarder_color=boarder_color,
        )
        if save_fig:
            fig.savefig("models/Button/{}/policy_actions.pdf".format(base), dpi=300)


def plot_policy(baselines: list, env_id: str, save_fig: bool = False) -> None:
    """Plot policy actions for each state for Simple and Penalty environments"""
    if env_id == "Button":
        plot_policy_button(baselines, save_fig=save_fig)
    else:
        for base in baselines:
            mdp_q_table = None
            if base in ["q_monitor_sequential", "q_monitor_joint"]:
                mdp_q_table = np.load("models/{}/{}/mdp_q_table_1.npy".format(env_id, base))
                q_table = np.load("models/{}/{}/monitor_q_table_1.npy".format(env_id, base))
            else:
                q_table = np.load("models/{}/{}/critic_q_table_1.npy".format(env_id, base))
            policy_states, policy_actions = get_policy_states_actions(env_id, base, q_table, mdp_q_table)
            boarder_color = "r" if base in ["q_mdp", "zero_reward", "zero_reward_0"] else "green"
            fig = plot_env_actions(env_id, policy_states, policy_actions, boarder_color=boarder_color)
            if save_fig:
                fig.savefig("models/{}/{}/policy_actions.pdf".format(env_id, base), dpi=300)


# pylint: disable=too-many-locals
def get_policy_states_actions(
        env_id: str,
        baseline: str,
        q_table: np.ndarray,
        mdp_q_table: np.ndarray = None,
) -> (np.ndarray, np.ndarray):
    """Get policy actions for each state from the q-table"""
    n_states = int(CELL_SIZE[0] * CELL_SIZE[1])
    n_actions = 4
    if env_id == "Button":
        states = np.zeros((n_states * 2, 2))
        states[:n_states, 0], states[n_states:, 1] = np.arange(n_states), np.arange(n_states)
        states[n_states:, 1] = 1

    else:
        states = np.arange(n_states)
    policy_actions = np.zeros((states.shape[0], 2))

    for i in range(states.shape[0]):
        state = get_state_ind(states[i], n_states) if isinstance(states[i], np.ndarray) else states[i]
        if q_table.shape[1] == n_actions:
            mdp_action = np.argmax(q_table[state])
            mon_action = 0
        else:
            if mdp_q_table is None:
                mdp_action, mon_action = ind_to_action(np.argmax(q_table[state]))
            else:
                mdp_action = np.argmax(mdp_q_table[state])
                if baseline == "q_monitor_sequential":
                    off_q, on_q = (
                        q_table[state, mdp_action],
                        q_table[state, mdp_action + n_actions],
                    )
                    mon_action = 0 if off_q > on_q else 1
                else:
                    q_value = np.concatenate((mdp_q_table[state], mdp_q_table[state])) + q_table[state]
                    mdp_action, mon_action = ind_to_action(np.argmax(q_value))
        policy_actions[i] = [mdp_action, mon_action]
    return states, policy_actions


def mean_time_period(vec: np.ndarray, period: int) -> np.ndarray:
    """Calculate average reward over a period of time"""
    mean_vec = np.zeros(len(vec) // period)
    for i in range(len(vec) // period):
        mean_vec[i] = np.mean(vec[i * period : i * period + period])
    return mean_vec


def discount_episode_reward(reward: dict, gamma: float = 0.99) -> (np.ndarray, np.ndarray):
    """Calculate each episode discount reward and number of timesteps"""
    discount = [gamma**i for i in range(100)]
    discount_reward, length = np.zeros(len(reward)), np.zeros(len(reward))
    for i, key in enumerate(reward.keys()):
        length[i] = len(reward[key])
        discount_reward[i] = np.sum(np.array(reward[key]) * np.array(discount[: len(reward[key])]))
    return discount_reward, length


def calculate_confidence_interval(vector: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Calculate confidence interval for 2d numpy array"""
    if confidence > 1 or confidence < 0:
        raise ValueError("the confidence value should be between [0, 1]")
    results = np.zeros(vector.shape[1])
    for i in range(vector.shape[1]):
        results[i] = scipy.stats.sem(vector[:, i]) * scipy.stats.t.ppf((1 + confidence) / 2.0, len(vector[:, i]) - 1)
    return results


def ind_to_action(action_ind: int, n_actions: int = 4) -> (int, int):
    """Transform integer index to MDP and Monitor actions"""
    mon_action, mdp_action = action_ind // n_actions, action_ind % n_actions
    return mdp_action, mon_action


def get_state_ind(state: list, n_states: int) -> int:
    """Transform MDP and Monitor states to an index integer"""
    return int(state[1] * n_states + state[0])


def sum_ep_timesteps(ep_timesteps: np.ndarray) -> np.ndarray:
    """add number of timesteps for each episode"""
    sum_timesteps = np.zeros(len(ep_timesteps))
    last_length = 0
    for i, timestep in enumerate(ep_timesteps):
        sum_timesteps[i] = last_length + timestep
        last_length = sum_timesteps[i]
    return sum_timesteps


def plot_arrows(env_id: str, states: np.ndarray, actions: np.ndarray, shift_arrow: int) -> None:
    """Plot arrows in env"""
    arrows_shift = shift_arrow * 0.2
    for i, state_i in enumerate(states):
        x_shift, y_shift = 0, 0
        state = state_i[0] if isinstance(states[i], np.ndarray) else state_i
        if state == 2:  # skip the goal state
            continue
        pos = np.array([state // CELL_SIZE[0], state % CELL_SIZE[0]])  # MDP state
        pos[0] = np.abs(pos[0] - CELL_SIZE[0] + 1)
        if shift_arrow != 0:
            if actions[i, 0] in [0, 2]:  # left and right MDP actions
                x_shift = arrows_shift
            else:
                y_shift = arrows_shift
        if env_id == "Button":
            line_c = "r" if state_i[1] == 0 else "b"  # Monitor action
        else:
            line_c = "r" if actions[i, 1] == 0 else "b"  # Monitor action
        arrow = ARROWS[actions[i, 0]]
        plt.arrow(
            pos[1] + y_shift,
            pos[0] + x_shift,
            SCALE * arrow[0],
            SCALE * arrow[1],
            lw=2.5,
            head_length=0.2,
            head_width=0.15,
            color=line_c,
            linestyle="-" if shift_arrow == 0 else "--",
        )


# pylint: disable=too-many-arguments, too-many-locals
def plot_env_actions(
        env_id: str,
        states,
        actions,
        legend: bool = False,
        plot_both: bool = False,
        boarder_color: str = "black",
):
    """Plot Simple, Penalty, and Button env in grid world"""
    # load images
    fire_img = plt.imread("img/fire_img.png")
    agent_img = plt.imread("img/agent_img.png")
    gold_img = plt.imread("img/gold_img.png")
    button_img = plt.imread("img/button_img.png")
    shift = SCALE * 2
    fig = plt.figure(figsize=CELL_SIZE)
    plt.hlines(np.arange(CELL_SIZE[1] + 1) - shift, -shift, CELL_SIZE[0] - shift, color="black")
    plt.vlines(np.arange(CELL_SIZE[0] + 1) - shift, -shift, CELL_SIZE[1] - shift, color="black")

    # outboard color
    if boarder_color != "black":
        plt.hlines(-shift, -shift, CELL_SIZE[0] - shift + 1, lw=7, color=boarder_color)
        plt.hlines(CELL_SIZE[1] - shift, -shift, CELL_SIZE[0] - shift + 1, lw=7, color=boarder_color)
        plt.vlines(-shift, -shift, CELL_SIZE[1] - shift, lw=7, color=boarder_color)
        plt.vlines(CELL_SIZE[0] - shift, -shift, CELL_SIZE[1] - shift, lw=7, color=boarder_color)

    if plot_both:
        plot_arrows(env_id, states["off"], actions["off"], shift_arrow=-1)
        plot_arrows(env_id, states["on"], actions["on"], shift_arrow=1)
    else:
        plot_arrows(env_id, states, actions, shift_arrow=0)

    if legend:
        mon_off = matplotlib.patches.Patch(color="r", label="Monitor Off")
        mon_on = matplotlib.patches.Patch(color="b", label="Monitor On")
        plt.legend(handles=[mon_off, mon_on], fontsize=7, loc=(0.28, 1.0))
    plt.xlim(-shift, CELL_SIZE[0] - shift)
    plt.ylim(-shift, CELL_SIZE[1] - shift)
    plt.axis("off")
    if env_id in ["Penalty", "Button"]:
        fire_1 = fig.add_axes([0.38, 0.7, 0.25, 0.25], anchor="NE", zorder=-1, alpha=0.2)
        fire_1.imshow(fire_img)
        fire_1.axis("off")

        fire_2 = fig.add_axes([0.38, 0.4, 0.25, 0.25], anchor="NE", zorder=-1, alpha=0.2)
        fire_2.imshow(fire_img)
        fire_2.axis("off")

    if env_id == "Button":
        # button = fig.add_axes([0.69, -0.09, 0.22, 0.22], anchor="NE", zorder=-1)  # cell 6
        # button = fig.add_axes([0.45, -0.09, 0.22, 0.22], anchor="NE", zorder=-1)  # cell 7
        button = fig.add_axes([0.69, -0.09, 0.22, 0.22], anchor="NE", zorder=-1, alpha=0.2)  # cell 8
        button.imshow(button_img)
        button.axis("off")

    agent = fig.add_axes([0.12, 0.77, 0.175, 0.175], anchor="NE", zorder=-1, alpha=0.2)
    agent.imshow(agent_img)
    agent.axis("off")

    gold = fig.add_axes([0.7, 0.72, 0.2, 0.2], anchor="NE", zorder=-1, alpha=0.2)
    gold.imshow(gold_img)
    gold.axis("off")
    fig.tight_layout()
    return fig


# pylint: disable=too-many-arguments, too-many-locals
def plot_train_joint_reward_timesteps(
        env_id: str,
        baselines: list,
        n_seeds: int = 30,
        timesteps_freq: int = 10000,
        plot_mean: bool = False,
        save_fig: bool = False,
) -> None:
    """Plot joint reward Vs number of timesteps with different random seeds"""
    fig = plt.figure(figsize=(7, 4))
    for i, baseline in enumerate(baselines):
        all_y_axis = []
        for seed in range(n_seeds):
            train_reward = np.load(
                "models/{}/{}/training_joint_reward_{}.npy".format(env_id, baseline, seed),
                allow_pickle=True,
            )[()]
            ep_reward, ep_length = discount_episode_reward(train_reward)
            ep_timesteps_sum = sum_ep_timesteps(ep_length)
            x_axis = np.arange(ep_length[0], int(ep_timesteps_sum[-1]), timesteps_freq)

            y_axis = np.ones(len(x_axis))
            count = 0
            for timestep in range(int(ep_length[0]), int(ep_timesteps_sum[-1]) + 1, timesteps_freq):
                if timestep > N_TRAIN_TIMESTEPS:
                    break
                y_axis[count] = ep_reward[np.where(ep_timesteps_sum > timestep)[0][0]]
                count += 1
            y_axis[-1] = y_axis[-2]
            all_y_axis.append(y_axis)
            label = BASELINES[baseline] if seed == 0 else None
            if not plot_mean:
                plt.plot(x_axis, y_axis, lw=1, c=COLORS[i], alpha=ALPHAS[i], label=label)
                plt.scatter(x_axis, y_axis, marker="*", s=5, c=COLORS[i], alpha=ALPHAS[i])
        if plot_mean:
            min_len = np.min([len(i) for i in all_y_axis])
            all_y_axis = np.array([i[:min_len] for i in all_y_axis]).reshape(n_seeds, min_len)
            x_axis = x_axis[:min_len]
            mean_reward = np.mean(np.array(all_y_axis), 0)
            confi_reward = calculate_confidence_interval(np.array(all_y_axis))
            plt.plot(x_axis, mean_reward, lw=2, c=COLORS[i], alpha=ALPHAS[i], label=BASELINES[baseline])
            plt.errorbar(x_axis, mean_reward, yerr=confi_reward, elinewidth=1, capsize=2, c=COLORS[i], alpha=ALPHAS[i])

    plt.xlabel("Training Timesteps", fontsize=12)
    if env_id == "Simple":
        plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(1, 4))
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if env_id == "Button":
        plt.legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    if save_fig:
        fig.savefig(
            "models/{}/cheat_timesteps_seeds_training_joint_rewar{}.pdf".format(env_id, "d_mean" if plot_mean else "d"),
            dpi=300,
        )
