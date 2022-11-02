import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

sns.set_theme(style="whitegrid")
# np.random.seed(42)

#2 player game
k = 2
n_agents = 2  # number of agents
n_iters = 100
strategy = "CM"

f = 0.9 #random.randrange(0,1) #recency parameter
e = 0.5 #random.ranrange(0,2) #experimentation parameter, take new actions
epsilon = 0.5
w = 10

rew = defaultdict(list)
rew["CC"] = [2, 5]
rew["CM"] = [0, 0]
rew["MC"] = [0, 0]
rew["MM"] = [5, 2]

def error_bars(x, w):
    #error bar for moving average
    return np.sqrt(np.convolve(np.array(x)**2, np.ones(w), 'valid') / w - moving_average(np.array(x), w)**2)
    # return np.std(x, axis=0)/np.sqrt(x.shape[0])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_reward(agent, r_choice):
    return rew[r_choice][agent]


def propensity_update(agent_propensity, actions):

    for agent in range(n_agents):
        a_list = list(actions)
        alt_act = strategy.replace(actions[agent], "")
        a_list[agent] = alt_act
        alt_strat = "".join(a_list)
        agent_propensity[agent][choice] = (1-f)*agent_propensity[agent][choice] + get_reward(agent, actions)*(1-e)
        agent_propensity[agent][alt_act] = (1-f)*agent_propensity[agent][alt_act] + get_reward(agent, alt_strat)*e/(k-1)
        if agent_propensity[agent][choice]<0 or agent_propensity[agent][alt_act]<0:
            print("prop error")
    return agent_propensity

def get_choice(agent_propensity):
    total = agent_propensity["C"] + agent_propensity["M"] + 0.0001
    prop_list = [agent_propensity["C"] / total, agent_propensity["M"] / total]
    maxval = max(prop_list)
    maxind = prop_list.index(maxval)
    return strategy[maxind]

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 1)
    _, ax1 = plt.subplots(1, 1)

    axs.set_xlabel('Iterations')
    axs.set_ylabel('Reward')
    # axs.set_title("n="+str(n_agents)+ "k="+str(k)+ "b="+str(b))
    name = "2-player"
    markers = ['o', 'v', 's', 'p', 'P', '*', 'X', 'D', 'd', 'x']
    #initialize
    agent_propensity = defaultdict(lambda: defaultdict(int))
    agent_choice = np.chararray((n_agents, n_iters))
    agent_reward = np.zeros((n_agents,  n_iters))
    temperature = 0

    df_t = pd.DataFrame()

    strategies = []
    df_list = []

    df = pd.DataFrame()

    for iter in range(n_iters):
        action = ""
        choices = []
        temperature = temperature + iter

        for agent in range(n_agents):
            if agent == 0:
                choice = 'C' if np.random.rand() < 0.5 else 'M'
            if agent == 1:
                if temperature < (n_iters - n_iters/4):
                    choice = strategy[np.random.randint(2)] if np.random.rand() < epsilon else get_choice(agent_propensity[agent])
                else:
                    choice = get_choice(agent_propensity[agent])
            choices.append(choice)
        action = action + choices[0]+choices[1]
        if temperature < (n_iters - n_iters / 4):
            agent_propensity = propensity_update(agent_propensity, action)
        for agent in range(n_agents):
            agent_reward[agent, iter] = get_reward(agent, action)
            agent_choice[agent, iter] = action[agent]

        # print([(elfarol.xk > elfarol.b).sum(), (elfarol.xk <= elfarol.b).sum()], ['Attendance > b', 'Attendance <= b'])
    axs.plot(list(range(n_iters)), agent_reward[0,:], label="agent0")
    axs.plot(list(range(n_iters)), agent_reward[1,:], label="agent1")
    # print("average {0} reward : {1}".format(name, elfarol.mean()))
    # moving_avg = moving_average(elfarol, w)
    # yerr = error_bars(elfarol, w)
    # axs.errorbar(list(range(len(elfarol)))[0::10], moving_avg[0::10], yerr=yerr[0::10], label="{0}".format(name), marker=markers[choice], capsize=3)  # , uplims=True, lolims=True)#alpha=0.1)
    axs.legend()

    # df['actions'] = list(range(n_iters*2))
    temperature = int(n_iters - n_iters/4)

    df['player'] = [0]*(n_iters-temperature) + [1]*(n_iters-temperature)
    df["actions"] = agent_choice[0,:][temperature:].tolist() + agent_choice[1,:][temperature:].tolist()
    # df_list.append(df)
    #
    # temperature = 0
    # df_tmp = pd.DataFrame.from_dict(day_wise_attendance_week, orient='index')
    # print(df_tmp.mean(axis=1))
    # df_t = pd.concat(df_list)
    # for i in range(3):
    # sns.histplot(data=df, x="actions", y="propensities", kind="bar", ax=ax1, hue="reward")
    sns.histplot(data=df, x="actions", hue="player", ax=ax1)#, multiple="layer", shrink=0.8, palette="Set2")
    # ax1.set_xlabel("Actions")
    # ax1.set_ylabel("Propensities")
    # ax1.title("n="+str(n_agents)+ "k="+str(k)+ "b="+str(b))
    plt.show()