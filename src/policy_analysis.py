"""Plotting functions for the training/testing, heatmaps for the q-tables, policy actions, and final trajectories"""
import colorsys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import scipy
import seaborn as sns


ARROWS = {0: (-1.5, 0), 1: (0, -1.5), 2: (1.5, 0), 3: (0, 1.5)}
STATES = np.arange(9)
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
COLORS = [
    "black",
    "y",
    "brown",
    "g",
    "b",
    "r",
]
ALPHAS = [0.5, 0.9, 0.9, 0.6, 0.4, 0.4]
N_TRAIN_EP = 10000
BASELINES = {
    "q_learning": "Q-Learning in MDP",
    "reward_model": "Reward Model",
    "q_monitor_joint": r"$Q_{joint}$",
    "q_monitor_sequential": r"$Q_{Sequential}$",
    "q_mdp": r"$Q_{ignore}$",
    "zero_reward": r"$Q_{\bot=0}$",
}


def set_blind_colors() -> tuple:
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
ALPHAS = np.ones(len(COLORS))


def plot_q_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Monitor Q-Table for values as a heatmap"""
    q_table = np.round(np.load(log_dir + "/critic_q_table_1.npy"), 2)
    y_sticks = JOINT_STATES if q_table.shape[0] == 18 else np.arange(q_table.shape[0])

    fig = plt.figure(figsize=(7, 8 if len(y_sticks) == 18 else 5))
    axis = sns.heatmap(q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
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
    mdp_q_table = np.round(np.load(log_dir + "/mdp_q_table_1.npy"), 2)
    mon_q_table = np.round(np.load(log_dir + "/monitor_q_table_1.npy"), 2)
    y_sticks = JOINT_STATES if mon_q_table.shape[0] == 18 else np.arange(mon_q_table.shape[0])

    fig = plt.figure(figsize=(4, 5))
    axis = sns.heatmap(mdp_q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    axis.set_xlabel("Actions", fontsize=15)
    axis.set_ylabel("States", fontsize=15)
    axis.set_xticklabels(MDP_ACTIONS, fontsize=13)
    axis.set_yticks(0.5 + np.arange(len(STATES)))
    axis.set_yticklabels(STATES, fontsize=13)
    fig.tight_layout()

    fig_1 = plt.figure(figsize=(7, 8 if len(y_sticks) == 18 else 5))
    axis_1 = sns.heatmap(mon_q_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    axis_1.set_xlabel("Actions", fontsize=15)
    axis_1.set_ylabel("States", fontsize=15)
    axis_1.set_xticklabels(MDP_ACTIONS if mon_q_table.shape[1] == 4 else JOINT_ACTIONS, fontsize=13)
    axis_1.set_yticks(0.5 + np.arange(len(y_sticks)))
    axis_1.set_yticklabels(y_sticks, fontsize=10 if len(y_sticks) == 18 else 13)
    fig_1.tight_layout()

    if save_fig:
        fig.savefig(log_dir + "/mdp_q_table_heatmap.pdf", dpi=300)
        fig_1.savefig(log_dir + "/monitor_q_table_heatmap.pdf", dpi=300)


def plot_reward_table_heatmap(log_dir: str, save_fig: bool = False) -> None:
    """Plot Predictive reward table values as a heatmap"""
    fig = plt.figure(figsize=(4, 5))
    r_table = np.round(np.load(log_dir + "/reward_model_table_1.npy"), 2)
    ax_r = sns.heatmap(r_table, cmap="crest", annot=True, linewidth=0.1, fmt="g", annot_kws={"fontsize": 12})
    ax_r.set_xlabel("Actions", fontsize=15)
    ax_r.set_ylabel("States", fontsize=15)
    ax_r.set_xticklabels(MDP_ACTIONS, fontsize=15)
    ax_r.set_yticklabels(np.arange(r_table.shape[0]), fontsize=15)
    fig.tight_layout()
    if save_fig:
        fig.savefig(log_dir + "/predictive_reward_table_heatmap.pdf", dpi=300)


# pylint: disable=too-many-arguments, too-many-locals
def plot_policy_trajectory(
        log_dir: str,
        env_name: str,
        cell_size: tuple = (3, 3),
        scale: float = 0.25,
        traj_n: int = 1,
        save_fig: bool = False,
) -> None:
    """
    plot final actions actions as arrows in grid environment
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
        on_fig = plot_env_actions(env_name.split("-")[1], on_states, on_actions, cell_size, scale)
        off_fig = plot_env_actions(env_name.split("-")[1], off_states, off_actions, cell_size, scale)
        if save_fig:
            on_fig.savefig(log_dir + "/final_policy_performance_on.pdf", dpi=300)
            off_fig.savefig(log_dir + "/final_policy_performance_off.pdf", dpi=300)
    else:
        states, actions = traj[traj_n]["states"], traj[traj_n]["actions"]
        fig = plot_env_actions(env_name.split("-")[1], states, actions, cell_size, scale)
        # fig.tight_layout()
        if save_fig:
            fig.savefig(log_dir + "/final_policy_performance.pdf", dpi=300)


# pylint: disable=too-many-locals
def plot_policy_switch(baselines: list, cell_size: tuple = (3, 3), scale: float = 0.25, save_fig: bool = False) -> None:
    """Plot policy actions for each state in switch environment"""
    n_states = int(cell_size[0] * cell_size[1])
    for base in baselines:
        mdp_q_table = None
        if base in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q_table = np.load("models/Switch/{}/mdp_q_table.npy".format(base))
            q_table = np.load("models/Switch/{}/monitor_q_table.npy".format(base))
        else:
            q_table = np.load("models/Switch/{}/critic_q_table.npy".format(base))
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
        on_fig = plot_env_actions("Switch", states[n_states:], on_actions, cell_size, scale)
        off_fig = plot_env_actions("Switch", states[:n_states], off_actions, cell_size, scale)
        if save_fig:
            on_fig.savefig("models/Switch/{}/policy_actions_on.pdf".format(base), dpi=300)
            off_fig.savefig("models/Switch/{}/policy_actions_off.pdf".format(base), dpi=300)


def plot_policy(baselines: list, env_id: str, cell_size: tuple = (3, 3), scale: float = 0.25, save_fig: bool = False):
    """Plot policy actions for each state for Simple and Fire environments"""
    if env_id == "Switch":
        raise NotImplemented

    for i, base in enumerate(baselines):
        mdp_q_table = None
        if base in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q_table = np.load("models/{}/{}/mdp_q_table.npy".format(env_id, base))
            q_table = np.load("models/{}/{}/monitor_q_table.npy".format(env_id, base))
        else:
            q_table = np.load("models/{}/{}/critic_q_table.npy".format(env_id, base))
        policy_states, policy_actions = get_policy_states_actions(env_id, base, q_table, mdp_q_table)
        fig = plot_env_actions(env_id, policy_states, policy_actions, cell_size, scale)
        if save_fig:
            fig.savefig("models/{}/{}/policy_actions.pdf".format(env_id, base), dpi=300)


# pylint: disable=too-many-locals
def get_policy_states_actions(
        env_id: str,
        baseline: str,
        q_table: np.ndarray,
        mdp_q_table: np.ndarray = None,
        cell_size: tuple = (3, 3),
) -> tuple:
    """Get policy actions for each state from the q-table"""
    n_states = int(cell_size[0] * cell_size[1])
    n_actions = 4
    if env_id == "Switch":
        states = np.zeros((n_states * 2, 2))
        states[:n_states, 0], states[n_states:, 1] = np.arange(n_states), np.arange(n_states)
        states[n_states:, 1] = 1

    else:
        states = np.arange(n_states)
    policy_actions = np.zeros((states.shape[0], 2))

    for i in range(states.shape[0]):
        if isinstance(states[i], np.ndarray):
            state = get_state_ind(states[i], n_states)
        else:
            state = states[i]
        if q_table.shape[1] == n_actions:
            mdp_action = np.argmax(q_table[state])
            mon_action = 0
        else:
            if mdp_q_table is None:
                mdp_action, mon_action = ind_to_action(np.argmax(q_table[state]))
            else:
                mdp_action = np.argmax(mdp_q_table[state])
                if baseline == "q_monitor_sequential":
                    off_q, on_q = q_table[state, mdp_action], q_table[state, mdp_action + n_actions]
                    mon_action = 0 if off_q > on_q else 1
                else:
                    q_value = np.concatenate((mdp_q_table[state], mdp_q_table[state])) + q_table[state]
                    mdp_action, mon_action = ind_to_action(np.argmax(q_value))
        policy_actions[i] = [mdp_action, mon_action]
    return states, policy_actions


def calculate_confidence_interval(vector: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Calculate confidence interval for 2d numpy array"""
    if confidence > 1 or confidence < 0:
        raise ValueError("the confidence value should be between [0, 1]")
    results = np.zeros(vector.shape[1])
    for i in range(vector.shape[1]):
        results[i] = scipy.stats.sem(vector[:, i]) * scipy.stats.t.ppf((1 + confidence) / 2.0, len(vector[:, i]) - 1)
    return results


def get_action_ind(action: list, n_actions: int = 4) -> int:
    """Transform MDP and Monitor actions to an index integer"""
    mdp_action, mon_action = action[0], action[1]
    return int(mon_action * n_actions + mdp_action)


def ind_to_action(action_ind: int, n_actions: int = 4) -> tuple:
    """Transform integer index to MDP and Monitor actions"""
    mon_action, mdp_action = action_ind // n_actions, action_ind % n_actions
    return mdp_action, mon_action


def get_state_ind(state: list, n_states: int) -> int:
    """Transform MDP and Monitor states to an index integer"""
    return int(state[1] * n_states + state[0])


def ind_to_state(state_ind: int, n_states: int, n_mon_states: int = 2) -> tuple:
    """Transform integer index to MDP and Monitor states"""
    if state_ind >= n_states * n_mon_states:
        raise ValueError("State index is larger than max Number of states")
    mon_state, mdp_state = state_ind // n_states, state_ind % n_states
    return mdp_state, mon_state


# pylint: disable=too-many-locals
def plot_env_actions(env_id: str, states, actions, cell_size: tuple = (3, 3), scale: float = 0.25):
    """Plot Simple, Fire, and Switch env in grid world"""
    # load images
    fire_img = plt.imread("img/fire_img.png")
    agent_img = plt.imread("img/agent_img.png")
    gold_img = plt.imread("img/gold_img.png")
    switch_img = plt.imread("img/switch_img.png")
    shift = scale * 2

    fig = plt.figure(figsize=cell_size)
    plt.hlines(np.arange(cell_size[1] + 1) - shift, -shift, cell_size[0] - shift, color="black")
    plt.vlines(np.arange(cell_size[0] + 1) - shift, -shift, cell_size[1] - shift, color="black")

    if len(states) == 18:
        raise NotImplemented
    for i in range(len(states)):
        if states[i, 0] == 2:  # skip the gaol state
            continue
        pos = np.array([states[i, 0] // cell_size[0], states[i, 0] % cell_size[0]])  # MDP state
        pos[0] = np.abs(pos[0] - cell_size[0] + 1)
        if env_id == "Switch":
            line_c = "r" if states[i, 1] == 0 else "b"  # Monitor action
        else:
            line_c = "r" if actions[i, 1] == 0 else "b"  # Monitor action
        arrow = ARROWS[actions[i, 0]]
        plt.arrow(
            pos[1],
            pos[0],
            scale * arrow[0],
            scale * arrow[1],
            lw=1.8,
            head_length=0.1,
            head_width=0.15,
            color=line_c,
        )

    mon_off = matplotlib.patches.Patch(color="r", label="Monitor Off")
    mon_on = matplotlib.patches.Patch(color="b", label="Monitor On")
    plt.legend(handles=[mon_off, mon_on], fontsize=7, loc=(0.28, 1.0))
    plt.xlim(-shift, cell_size[0] - shift)
    plt.ylim(-shift, cell_size[1] - shift)
    plt.axis("off")
    if env_id in ["Fire", "Switch"]:
        fire_1 = fig.add_axes([0.41, 0.67, 0.2, 0.2], anchor="NE", zorder=-1)
        fire_1.imshow(fire_img)
        fire_1.axis("off")

        fire_2 = fig.add_axes([0.41, 0.41, 0.2, 0.2], anchor="NE", zorder=-1)
        fire_2.imshow(fire_img)
        fire_2.axis("off")

    if env_id == "Switch":
        switch = fig.add_axes([0.12, 0.1, 0.15, 0.15], anchor="NE", zorder=-1)
        switch.imshow(switch_img)
        switch.axis("off")

    agent = fig.add_axes([0.14, 0.75, 0.12, 0.12], anchor="NE", zorder=-1)
    agent.imshow(agent_img)
    agent.axis("off")

    gold = fig.add_axes([0.7, 0.7, 0.15, 0.15], anchor="NE", zorder=-1)
    gold.imshow(gold_img)
    gold.axis("off")

    return fig


def mean_time_period(vec: np.ndarray, period: int) -> np.ndarray:
    """Calculate average reward over a period of time"""
    mean_vec = np.zeros(len(vec) // period)
    for i in range(len(vec) // period):
        mean_vec[i] = np.mean(vec[i * period: i * period + period])
    return mean_vec


def discount_episode_reward(reward: dict, gamma: float = 0.99) -> (np.ndarray, np.ndarray):
    """Calculate each episode discount reward and number of timesteps"""
    discount = [gamma**i for i in range(100)]
    discount_reward, length = np.zeros(len(reward)), np.zeros(len(reward))
    for i, key in enumerate(reward.keys()):
        length[i] = len(reward[key])
        discount_reward[i] = np.sum(np.array(reward[key]) * np.array(discount[:len(reward[key])]))
    return discount_reward, length


# pylint: disable=too-many-locals
def plot_joint_reward(
        baselines: list,
        env_name: str,
        testing_freq: int = 10,
        save_fig: bool = False,
) -> None:
    """Plot Episode joint reward per episode"""
    train_freq = 100
    joint_rewards = np.zeros((len(baselines), N_TRAIN_EP))
    eval_joint_rewards = np.zeros((len(baselines), N_TRAIN_EP // testing_freq))
    x_axis = testing_freq * np.arange(N_TRAIN_EP // testing_freq)
    for i, base in enumerate(baselines):
        joint_rewards[i], _ = discount_episode_reward(
            np.load("models/{}/{}/training_joint_reward.npy".format(env_name, base), allow_pickle=True)[()]
        )
        eval_joint_rewards[i] = np.mean(np.load("models/{}/{}/evaluation_joint_reward.npy".format(env_name, base)), 1)
        eval_joint_rewards[i, -1] = eval_joint_rewards[i, -2]

    fig = plt.figure(figsize=(7, 4))
    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)
    for i, reward in enumerate(joint_rewards):
        mean_reward = mean_time_period(reward, train_freq)
        plt.plot(train_x_axis, mean_reward, lw=2, alpha=ALPHAS[i], color=COLORS[i], label=BASELINES[baselines[i]])
        plt.plot(train_x_axis, mean_time_period(reward, train_freq), "*", alpha=ALPHAS[i], color=COLORS[i])
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
    """Plot joint reward Vs number of episodes with different random seeds"""
    n_seeds = 30
    train_freq = 100
    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)
    joint_reward = np.zeros((n_seeds, N_TRAIN_EP // train_freq))

    fig = plt.figure(figsize=(7, 4))
    for seed in range(n_seeds):
        reward = np.load(
            "models/{}/{}/training_joint_reward_{}.npy".format(env_name, baseline, seed),
            allow_pickle=True,
        )[()]
        joint_reward[seed], _ = mean_time_period(discount_episode_reward(reward), train_freq)
        plt.plot(train_x_axis, joint_reward[seed], lw=0.1)
        plt.scatter(train_x_axis, joint_reward[seed], marker="*", s=1)

    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_fig:
        fig.savefig("models/{}/{}/seeds_training_joint_reward.pdf".format(env_name, baseline), dpi=300)


# pylint: disable=too-many-locals
def plot_joint_reward_seeds_baselines(env_name: str, baselines: list, plot_mean: bool = False, save_fig: bool = False):
    """Plot joint reward Vs number of episodes with different random seeds for Baselines"""
    n_seeds = 30
    train_freq = 500
    train_x_axis = train_freq * np.arange(N_TRAIN_EP // train_freq)

    fig = plt.figure(figsize=(7, 4))
    for i, baseline in enumerate(baselines):
        all_rewards = []
        for seed in range(n_seeds):
            label = BASELINES[baselines[i]] if seed == 0 else None
            ep_reward, _ = discount_episode_reward(
                np.load(
                    "models/{}/{}/training_joint_reward_{}.npy".format(env_name, baseline, seed),
                    allow_pickle=True,
                )[()]
            )
            reward = mean_time_period(ep_reward, train_freq)
            all_rewards.append(reward)
            if not plot_mean:
                plt.plot(train_x_axis, reward, lw=1, c=COLORS[i], alpha=ALPHAS[i], label=label)
                plt.scatter(train_x_axis, reward, marker="*", s=2, c=COLORS[i], alpha=ALPHAS[i])

        if plot_mean:
            mean_reward = np.mean(np.array(all_rewards), 0)
            conf_reward = calculate_confidence_interval(np.array(all_rewards))
            plt.plot(train_x_axis, mean_reward, lw=3, c=COLORS[i], alpha=ALPHAS[i], label=BASELINES[baseline])
            # plt.scatter(train_x_axis, mean_reward, marker="*", s=20, c=COLORS[i], alpha=ALPHAS[i])
            plt.errorbar(
                train_x_axis, mean_reward, yerr=conf_reward, elinewidth=1, capsize=2, c=COLORS[i], alpha=ALPHAS[i]
            )
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig.savefig(
            "models/{}/seeds_training_joint_rewar{}.pdf".format(env_name, "d_mean" if plot_mean else "d"), dpi=300
        )


# pylint: disable=too-many-arguments, too-many-locals
def plot_train_joint_reward_timesteps(
        env_id: str,
        baselines: list,
        n_seeds: int = 30,
        timesteps_freq: int = 5000,
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
            plt.plot(x_axis, mean_reward, lw=3, c=COLORS[i], alpha=ALPHAS[i], label=BASELINES[baseline])
            # plt.scatter(x_axis, mean_reward, marker="*", s=20, c=COLORS[i], alpha=ALPHAS[i])
            plt.errorbar(x_axis, mean_reward, yerr=confi_reward, elinewidth=1, capsize=2, c=COLORS[i], alpha=ALPHAS[i])

    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Episode Joint Reward", fontsize=12)
    plt.grid(axis="y")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig.savefig(
            "models/{}/timesteps_seeds_training_joint_rewar{}.pdf".format(env_id, "d_mean" if plot_mean else "d"),
            dpi=300,
        )


def sum_ep_timesteps(ep_timesteps: np.ndarray) -> np.ndarray:
    """add number of timesteps for each episode"""
    sum_timesteps = np.zeros(len(ep_timesteps))
    last_length = 0
    for i, timestep in enumerate(ep_timesteps):
        sum_timesteps[i] = last_length + timestep
        last_length = sum_timesteps[i]
    return sum_timesteps
