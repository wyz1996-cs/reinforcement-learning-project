import matplotlib.pyplot as plt
import random
import numpy as np


# show final result
def plot_world(world, traps, start_position, goal, result=None):
    plt.figure(1)
    plt.ylim([0, len(world) - 1])
    plt.xlim([0, len(world) - 1])
    plt.xticks([i for i in range(len(world))], [str(i) for i in range(len(world))])
    plt.yticks([i for i in range(len(world))], [str(i) for i in range(len(world))])
    plt.grid()
    plt.scatter(start_position[0], start_position[1], s=150, color="blue", marker="o")
    plt.scatter(goal[0], goal[1], s=150, color="red", marker="s")
    for i in traps:
        plt.scatter(i[0], i[1], s=150, color="black", marker="H")
    if result != None:
        for i in range(len(result) - 1):
            plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], color="red", marker="*")
        plt.savefig("result.png", dpi=600)
        plt.show()
    else:
        plt.savefig("result.png", dpi=600)
        plt.show()



# return next state based on current state and action
def action_result(action, current_state):
    if action == "up":
        if current_state[1] == 11:
            return current_state
        else:
            return current_state[0], current_state[1] + 1
    elif action == "down":
        if current_state[1] == 0:
            return current_state
        else:
            return current_state[0], current_state[1] - 1
    elif action == "left":
        if current_state[0] == 0:
            return current_state
        else:
            return current_state[0] - 1, current_state[1]
    elif action == "right":
        if current_state[0] == 11:
            return current_state
        else:
            return current_state[0] + 1, current_state[1]


'''Reward function
if agent stay on current state, -5
if agent reach goal, +100
if agent step on traps, -10
if agent step on a grid is not goal or trap, -1
'''


def get_reward(state, goal, traps, current_state):
    if state == current_state:
        return -3
    if state == goal:
        return 100
    elif state in traps:
        return -10
    else:
        return -1


# obtain max q value
def get_maxq(qtable, state):
    temp = []
    for i in range(len(qtable[state[0]][state[1]])):
        temp.append(qtable[state[0]][state[1]][i])
    maxone = max(temp)
    argmax = np.argmax(temp)
    return maxone, argmax

# set up grid map where world[i][j]=(i,j)
world = [[(i, j) for j in range(12)] for i in range(12)]
traps = [(8, 11), (8, 10), (8, 9), (8, 8), (9, 8), (10, 8), (10, 9), (10, 10), (9, 5), (10, 7), (9, 2), (9, 7), (6, 9),(2, 2), (2, 1), (2, 0),(1,2)]
start_position = (1, 1)
goal_position = (9, 9)


# actions
action = ["up", "down", "left", "right"]

# set up 4*12*12 q table, for 12*12 grid, each grid has 4 actions, all values set to 0
# set up 12*12 policy table, all values set to 0
q_table = [[[0 for k in range(4)] for j in range(len(world))] for i in range(len(world))]
policy = [[0 for j in range(len(world))] for i in range(len(world))]

# learning parameters
episodes = 1000
alpha = 0.8
gamma = 0.7
epsilon = 0.5

# Learning
for episode in range(episodes):
    current_state = start_position
    while True:
        # for half change, it go through policy, another half take random action
        if random.randint(1, 100) / 100 > epsilon:
            next_actions = policy[current_state[0]][current_state[1]]
        else:
            next_actions = random.randint(0, 3)
        next_state = action_result(action[next_actions], current_state)
        reward = get_reward(next_state, goal_position, traps, current_state)

        # update q table and policy
        MAX_Q, _ = get_maxq(q_table, next_state)
        q_table[current_state[0]][current_state[1]][next_actions] += alpha * (reward + gamma * MAX_Q - q_table[current_state[0]][current_state[1]][next_actions])
        _, argmax = get_maxq(q_table, current_state)
        policy[current_state[0]][current_state[1]] = argmax


        current_state = next_state
        # if reward is 100, current episode ends
        if reward == 100 or reward == -10:
            break

# validate the result by going through the final policy
state = start_position
res = [state]

for i in range(200):
    actions = policy[state[0]][state[1]]
    next_state = action_result(action[actions], state)

    res.append(next_state)
    if next_state == goal_position:
        plot_world(world, traps, start_position, goal_position, res)
        break
    state = next_state
