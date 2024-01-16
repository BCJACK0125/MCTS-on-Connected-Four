"""
It uses the following classes and methods:

•	Node: This class represents a node in the search tree. It has the following attributes and methods:
    o	play: The action that leads to this node from its parent.
    o	parent: The parent node of this node.
    o	Q: The total reward of this node.
    o	N: The number of visits of this node.
    o	child: A dictionary that maps actions to child nodes.
    o	reward: The reward of this node for the current player.
    o	add_child(self, children: dict) -> None: This method adds a list of child nodes to the current node.
    o	value(self, explore_value: float = math.sqrt(2)): This method returns the value of this node according to the UCT formula. It takes an optional parameter explore_value that controls the exploration-exploitation trade-off.

•	MCTS: This class implements the MCTS algorithm. It has the following attributes and methods:
    o	State_root: The initial state of the game environment.
    o	root: The root node of the search tree.
    o	time: The time spent on the search.
    o	N_rollout: The number of rollouts performed during the search.
    o	node_select(self) -> tuple: This method selects a node to expand or simulate according to the UCT values. It returns a tuple of the selected node and the corresponding game state.
    o	exp(self, parent: Node, state: Environment) -> bool: This method expands a node by creating its child nodes based on the legal actions in the game state. It returns True if the node is expanded, False otherwise.
    o	roll(self, state: Environment) -> int: This method simulates a random playout from a given game state until the game is over. It returns the reward of the game outcome for the current player.
    o	back_propagation(self, node: Node, turn: int, value: int) -> None: This method updates the Q and N values of the nodes along the path from the given node to the root node. It takes the node, the current player’s turn, and the game outcome value as parameters.
    o	search(self, time_limit: int): This method performs the MCTS search for a given time limit. It repeatedly calls the node_select, roll, and back_propagation methods until the time limit is reached.
    o	best_play(self): This method returns the best action to take according to the search results. It chooses the child node with the highest average reward (Q/N) as the best node, and returns its play attribute.
    o	play(self, play): This method updates the root node and the game state according to a given action. It moves the root node to the corresponding child node, or creates a new root node if the action is not in the child dictionary.
    o	summary(self) -> tuple: This method returns a tuple of the number of rollouts and the time spent on the search.
"""

from Environment import Environment
from Parameters import Parameters
from MCTS import MCTS

import random, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from copy import deepcopy
import time
import math

class Parameters:
    Players = {'none': 0, 'green': 1, 'blue': 2}
    Rewards = {'green': 1, 'blue': 2, 'draw': 3}
    INF = float('inf')
    N_rows = 6
    N_columns = 7

class Environment:
    def __init__(self):
        self.status = [[0] * Parameters.N_columns for _ in range(Parameters.N_rows)]
        self.status = [[0, 2, 1, 1, 2, 1, 0],
                    [0, 2, 2, 1, 1, 2, 0],
                    [0, 1, 1, 1, 2, 1, 0],
                    [0, 2, 2, 2, 1, 2, 0],
                    [0, 2, 1, 1, 2, 1, 0],
                    [1, 2, 2, 2, 1, 1, 2]]
        self.who = Parameters.Players['blue']
        self.depth = [Parameters.N_rows - 1] * Parameters.N_columns
        self.depth = [4, 0, 0, 0, 0, 0, 4]
        self.record = []
        self.reward = [0, 4]
        self.flag = 0
        self.prob = np.zeros(Parameters.N_columns)
        self.best_play = None

    def get_status(self):
        return deepcopy(self.status)

    def play(self, column):
        self.status[self.depth[column]][column] = self.who
        self.record = [self.depth[column], column]
        self.depth[column] -= 1
        self.who = Parameters.Players['blue'] if self.who == Parameters.Players['green'] else Parameters.Players['green']

    def legal_judge(self):
        return [column for column in range(Parameters.N_columns) if self.status[0][column] == 0]

    def win_check(self):
        if len(self.record) > 0 and self.win_checking(self.record[0], self.record[1]):
            return self.status[self.record[0]][self.record[1]]
        return 0

    def win_checking(self, row, column):
        player = self.status[row][column]
        
        consec = 1
        # Check horizontal
        tmp_row = row
        while tmp_row + 1 < Parameters.N_rows and self.status[tmp_row + 1][column] == player:
            consec += 1
            tmp_row += 1
        tmp_row = row
        while tmp_row - 1 >= 0 and self.status[tmp_row - 1][column] == player:
            consec += 1
            tmp_row -= 1
        if consec >= 4:
            return True

        # Check vertical
        consec = 1
        tmp_column = column
        while tmp_column + 1 < Parameters.N_columns and self.status[row][tmp_column + 1] == player:
            consec += 1
            tmp_column += 1
        tmp_column = column
        while tmp_column - 1 >= 0 and self.status[row][tmp_column - 1] == player:
            consec += 1
            tmp_column -= 1
        if consec >= 4:
            return True

        # Check diagonal
        consec = 1
        tmp_row = row
        tmp_column = column
        while tmp_row + 1 < Parameters.N_rows and tmp_column + 1 < Parameters.N_columns and self.status[tmp_row + 1][tmp_column + 1] == player:
            consec += 1
            tmp_row += 1
            tmp_column += 1
        tmp_row = row
        tmp_column = column
        while tmp_row - 1 >= 0 and tmp_column - 1 >= 0 and self.status[tmp_row - 1][tmp_column - 1] == player:
            consec += 1
            tmp_row -= 1
            tmp_column -= 1
        if consec >= 4:
            return True

        # Check anti-diagonal
        consec = 1
        tmp_row = row
        tmp_column = column
        while tmp_row + 1 < Parameters.N_rows and tmp_column - 1 >= 0 and self.status[tmp_row + 1][tmp_column - 1] == player:
            consec += 1
            tmp_row += 1
            tmp_column -= 1
        tmp_row = row
        tmp_column = column
        while tmp_row - 1 >= 0 and tmp_column + 1 < Parameters.N_columns and self.status[tmp_row - 1][tmp_column + 1] == player:
            consec += 1
            tmp_row -= 1
            tmp_column += 1
        if consec >= 4:
            return True

        return False

    def over(self):
        return self.win_check() or len(self.legal_judge()) == 0

    def get_reward(self):
        if len(self.legal_judge()) == 0 and self.win_check() == 0:
            return Parameters.Rewards['draw']
        return Parameters.Rewards['green'] if self.win_check() == Parameters.Players['green'] else Parameters.Rewards['blue']

class Node:
    def __init__(self, play, parent):
        self.play = play
        self.parent = parent
        self.Q = 0
        self.N = 0
        self.child = {}
        self.reward = Parameters.Players['none']

    def add_child(self, children: dict) -> None:
        for child in children:
            self.child[child.play] = child

    def value(self, explore_value: float = math.sqrt(2)):
        if self.N == 0:
            return 0 if explore_value == 0 else Parameters.INF
        else:
            return self.Q / self.N + explore_value * math.sqrt(math.log(self.parent.N) / self.N)

class MCTS:
    def __init__(self, state=Environment()):
        self.State_root = deepcopy(state)
        self.root = Node(None, None)
        self.time = 0
        self.N_rollout = 0

    def node_select(self) -> tuple:
        node = self.root
        state = deepcopy(self.State_root)
        while len(node.child) != 0:
            children = node.child.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]
            node = random.choice(max_nodes)
            state.play(node.play)
            if node.N == 0:
                return node, state
        if self.exp(node, state):
            node = random.choice(list(node.child.values()))
            state.play(node.play)
        return node, state

    def exp(self, parent: Node, state: Environment) -> bool:
        if state.over():
            return False
        children = [Node(play, parent) for play in state.legal_judge()]
        parent.add_child(children)
        return True

    def roll(self, state: Environment) -> int:
        while not state.over():
            state.play(random.choice(state.legal_judge()))
        return state.get_reward()

    def back_propagation(self, node: Node, turn: int, value: int) -> None:
        # For the current player, not the next player
        reward = -1 if value == turn else 1
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if value == Parameters.Rewards['draw']:
                reward = 0
            else:
                reward *= -1
                

    def search(self, time_limit: int):
        start_time = time.process_time()
        N_rollout = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.node_select()
            value = self.roll(state)
            self.back_propagation(node, state.who, value)
            N_rollout += 1
        run_time = time.process_time() - start_time
        self.time = run_time
        self.N_rollout = N_rollout

    def best_play(self):
        if self.State_root.over():
            return -1
        # max_value = max(self.root.child.values(), key=lambda n: n.N).N
        # max_nodes = [n for n in self.root.child.values() if n.N == max_value]
        tmp = max(self.root.child.values(), key=lambda n: n.Q / n.N if n.N != 0 else n.N)
        max_value = tmp.Q / tmp.N if tmp.N != 0 else 0
        if max_value == 0:
            max_nodes = [n for n in self.root.child.values() if n.N == max_value]
        else:
            max_nodes = [n for n in self.root.child.values() if (n.N != 0 and n.Q / n.N == max_value)]
        best_child = random.choice(max_nodes)
        return best_child.play

    def play(self, play):
        if play in self.root.child:
            self.State_root.play(play)
            self.root = self.root.child[play]
            return
        self.State_root.play(play)
        self.root = Node(None, None)
    def summary(self) -> tuple:
        return self.N_rollout, self.time

def show(Env, MCTS_Agent, Parameters, flag):
    board = np.zeros((Parameters.N_rows, Parameters.N_columns))
    for row in range(Parameters.N_rows):
        for column in range(Parameters.N_columns):
            if Env.status[row][column] == 1:
                board[row][column] = 1  # Set color to green
            elif Env.status[row][column] == 2:
                board[row][column] = 2  # Set color to blue
    colors = ['gray', 'green', 'blue']
    # if Env.flag != 1:
    #     colors = ['gray', 'green', 'blue']
    # else:
    #     colors = ['gray', 'green', 'green']
    cmap = mcolors.ListedColormap(colors)
    plt.imshow(board, cmap=cmap)
    for i, prob in enumerate(Env.prob):
        if i == Env.best_play:
            plt.text(i, 6, "{:.2f}".format(prob), ha='center', va='center', fontsize=10, color='red')
        else:
            plt.text(i, 6, "{:.2f}".format(prob), ha='center', va='center', fontsize=10)
    plt.text(Env.record[1], Env.record[0], 'X', ha='center', va='center', fontsize=20) if len(Env.record) > 0 else None
    plt.xticks([])
    plt.yticks([])
    if MCTS_Agent.root.N == 0:
        win_rate = 0
    else:
        win_rate = MCTS_Agent.root.Q / MCTS_Agent.root.N
    if flag == 0:
        plt.title("Blue Win rate: {:.4f}".format(win_rate))
    else:
        plt.title("Green Win rate: {:.4f}".format(win_rate))
    plt.draw()
    if not os.path.exists("images"):
            os.mkdir("images")
    plt.savefig("images/" + str(Env.flag) + ".png")
    plt.pause(1)
    Env.flag += 1
    plt.clf()

def play():
    Env = Environment()
    MCTS_Agent = MCTS(Env)
    flag = 1
    while not Env.over():
        show(Env, MCTS_Agent, Parameters, flag)
        flag *= -1
        
        print("Thinking...")
        MCTS_Agent.search(8)
        rollouts, time = MCTS_Agent.summary()
        print("Statistics: ", rollouts, "rollouts in", time, "seconds")
        play = MCTS_Agent.best_play()
        print("Blue chose play: ", play)
        
        a = np.zeros(Parameters.N_columns)
        i = 0
        for child in MCTS_Agent.root.child.values():
            a[i] = child.value()
            i = 6
        Env.prob = a
        print("Blue Sibling values: ", a)
        Env.best_play = MCTS_Agent.best_play()
        print("Blue win rate: ", MCTS_Agent.root.child[play].value())
        
        Env.play(play)
        MCTS_Agent.play(play)
        
        if Env.over():
            print("Player Blue won!")
            flag = 1
            Env.flag == 3
            show(Env, MCTS_Agent, Parameters, flag)
            break
        
        random_agent = 0 if random.randint(0, 1) == 0 else 6
        print("Green chose play: ", random_agent)
        Env.play(random_agent)
        MCTS_Agent.play(random_agent)
        show(Env, MCTS_Agent, Parameters, flag)        
        a = np.zeros(Parameters.N_columns)
        i = 0
        for child in MCTS_Agent.root.child.values():
            a[i] = child.Q / child.N
            i = 6
        Env.prob = a
        print("Green Sibling values: ", a)
        Env.best_play = MCTS_Agent.best_play()
        print("Green win rate: ", MCTS_Agent.root.child[play].Q / MCTS_Agent.root.child[play].N)
        
        if Env.over():
            print("Player Green won!")
            flag = -1
            Env.flag == 3
            show(Env, MCTS_Agent, Parameters, flag)
            break

if __name__ == "__main__":
    while True:
        play()
        os.system("pause")
