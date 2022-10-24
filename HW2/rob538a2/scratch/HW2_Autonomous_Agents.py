# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:21:55 2022

@author: ananya
"""

import numpy as np
import ipdb 

BOARD_ROWS = 5
BOARD_COLS = 10
WIN_STATE = (3, 1) #T1
#LOSE_STATE = (1, 3)
START = (2, 3)
DETERMINISTIC = False


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        #self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self, state):
        if state == WIN_STATE:
            return 20
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal i.e. within the boundary of grid world
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 9):
                return nxtState
        return self.state

    def showBoard(self):
        self.board[self.state] = 20
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 20:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                out += token + ' | '
            print(out)
        print('-----------------')


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def chooseNextAction(self, next_position):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # current_position = self.State.state
                nxt_reward = self.Q_values[next_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd
        self.State.showBoard()
        

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                self.reset()
                break
            else:
                for a in self.actions:
                    n_state = self.State.nxtPosition(a)
                    reward = self.State.giveReward(n_state)
                    current_q_value = self.Q_values[self.State.state][a]
                    max_q_value = self.Q_values[n_state][self.chooseNextAction(n_state)]
                    new_q_value = current_q_value + self.lr * (reward + self.decay_gamma * max_q_value - current_q_value)
                    self.Q_values[self.State.state][a] = new_q_value
                    self.State = self.takeAction(a)
                    i += 1

    # def test(self):
    #             action = self.chooseAction()
    #             # append trace
    #             self.states.append([(self.State.state), action])
    #             print("current position {} action {}".format(self.State.state, action))
    #             # by taking the action, it reaches the next state
    #
    #             # mark is end
    #             self.State.isEndFunc()
    #             print("nxt state", self.State.state)
    #             print("---------------------")
    #             self.isEnd = self.State.isEnd


if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print("before play:")
    print(ag.Q_values)
    
    print("After play:")
    ag.play(5)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    
