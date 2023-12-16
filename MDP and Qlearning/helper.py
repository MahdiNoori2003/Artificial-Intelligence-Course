import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time


def table_plot_values(data, plot_label):

    fig, ax = plt.subplots()
    ax.axis('off')

    cmap = plt.cm.get_cmap('RdYlGn')
    heatmap = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha='center', va='center', color='black')

    cbar = fig.colorbar(heatmap)
    ax.set_title(plot_label)
    plt.show()


def table_plot_policy(data, plot_label):
    moves = moves = ["←", "↓", "→", "↑"]

    fig, ax = plt.subplots()
    ax.axis('off')

    heatmap = ax.imshow(data)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, moves[data[i, j]], ha='center',
                    va='center', color='black')

    ax.set_title(plot_label)
    plt.show()


def simulate_frozen_lake(agent):

    initial_state, _ = agent.reset()
    state = initial_state
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    img = ax.imshow(agent.env.render(), interpolation='nearest')

    while True:
        img.set_data(agent.env.render())
        display.display(fig)
        display.clear_output(wait=True)
        state, _, done = agent.take_action(agent.get_optimal_policy(state))
        time.sleep(0.8)
        if done:
            break

    img.set_data(agent.env.render())
    display.display(fig)
    display.clear_output(wait=True)
    plt.close(fig)


def plot_mean_values(rewards_with_reduction, rewards_without_reduction, execution_time_with_reduction, execution_time_without_reduction, title):
    EPISODES = rewards_with_reduction.shape[1]

    rewards_mean_with_reduction = np.mean(rewards_with_reduction, axis=0)
    execution_time_mean_with_reduction = np.mean(
        execution_time_with_reduction, axis=0)

    rewards_mean_without_reduction = np.mean(rewards_without_reduction, axis=0)
    execution_time_mean_without_reduction = np.mean(
        execution_time_without_reduction, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(EPISODES), rewards_mean_with_reduction,
             label="Reducing learning rate")
    ax1.plot(range(EPISODES), rewards_mean_without_reduction,
             label="Fixed learning rate")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Rewards per Episode')
    ax1.legend()
    ax1.set_xlim(0, 200)

    ax2.plot(range(EPISODES), execution_time_mean_with_reduction,
             label="Reducing learning rate")
    ax2.plot(range(EPISODES), execution_time_mean_without_reduction,
             label="Fixed learning rate")
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Execution Time')
    ax2.set_title('Mean Execution Time per Episode')
    ax2.legend()
    ax2.set_xlim(0, 200)

    plt.suptitle(title)

    plt.tight_layout()

    plt.show()


def simulate_taxi(agent):
    total_reward = 0
    initial_state, _ = agent.reset()
    state = initial_state
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    img = ax.imshow(agent.env.render(), interpolation='nearest')

    while True:
        img.set_data(agent.env.render())
        display.display(fig)
        display.clear_output(wait=True)
        state, reward, done = agent.take_action(
            agent.get_optimal_policy(state))
        total_reward += reward
        time.sleep(0.8)
        if done:
            break
    print(f"The resulting reward is {total_reward}")
    plt.close(fig)
