import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

sns.set_theme(style="whitegrid")
np.random.seed(42)

#El-Farol RL
b = 5  # attendance parameter
k = 6 #number of nights
n_agents = 42  # number of agents
n_weeks = 1000

f = 0.9 #random.randrange(0,1) #recency parameter
e = 0.5 #random.ranrange(0,2) #experimentation parameter, take new actions
epsilon = 0.5
w = 10

def error_bars(x, w):
    #error bar for moving average
    return np.sqrt(np.convolve(np.array(x)**2, np.ones(w), 'valid') / w - moving_average(np.array(x), w)**2)
    # return np.std(x, axis=0)/np.sqrt(x.shape[0])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_reward(attendance, r_choice):
    if r_choice==0:
        return attendance*np.exp(-attendance/b)
    if r_choice==1:
        return attendance@np.exp(-attendance/b)
    if r_choice==2:
        # if attendance @ np.exp(-attendance / b) - (attendance - 1) @ np.exp(-(attendance - 1) / b)<0:
        #     print("rew error")
        return abs(attendance*np.exp(-attendance/b) - (attendance-1)*np.exp(-(attendance-1)/b))

def propensity_update(agent_propensity, choice, attendance, r_choice):
    choice=int(choice)
    for night in range(k):
        if night == choice:
            agent_propensity[choice] = (1-f)*agent_propensity[choice] + get_reward(attendance, r_choice)*(1-e)
        else:
            agent_propensity[choice] = (1-f)*agent_propensity[choice] + get_reward(attendance, r_choice)*e/(k-1)
        if agent_propensity[choice]<0:
            print("prop error")
    return agent_propensity

def get_choice(agent_propensity):
    return np.argmax(agent_propensity/agent_propensity.sum())

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 1)
    _, ax1 = plt.subplots(1, 1)

    axs.set_xlabel('Week')
    axs.set_ylabel('System Reward')
    axs.set_title("n="+str(n_agents)+ "k="+str(k)+ "b="+str(b))
    name = "El-Farol"
    markers = ['o', 'v', 's', 'p', 'P', '*', 'X', 'D', 'd', 'x']
    #initialize
    agent_propensity = np.ones((n_agents, k))
    agent_choice = np.zeros((n_agents, n_weeks, k))
    agent_reward = np.zeros((n_agents,  n_weeks, k))
    elfarol = np.zeros((n_weeks))
    temperature = 0
    day_wise_attendance_week = defaultdict(list)
    df_t = pd.DataFrame()
    strategies = []
    df_list = []
    for r_choice in [0, 1, 2]:
        df = pd.DataFrame()
        for week in range(n_weeks):
            temperature = temperature + 1
            agent_attendances = np.zeros(k)
            for agent in range(n_agents):
                if temperature < (n_weeks - n_weeks/4):
                    choice = np.random.randint(k) if np.random.rand() < epsilon else get_choice(agent_propensity[agent])
                else:
                    choice = get_choice(agent_propensity[agent])
                agent_choice[agent, week, choice] += 1
                agent_attendances[choice] += 1
            for night in range(k):
                attendance = agent_choice[:, week, night].sum()
                day_wise_attendance_week[night].append(attendance)
                for agent in range(n_agents):
                    if r_choice == 0:
                        agent_propensity[agent] = propensity_update(agent_propensity[agent], np.where(agent_choice[agent,week,:]==1)[0][0], attendance, r_choice)
                        agent_reward[agent, week, night] = get_reward(attendance, r_choice)
                    elif r_choice == 1:
                        agent_propensity[agent] = propensity_update(agent_propensity[agent], np.where(agent_choice[agent,week,:]==1)[0][0], agent_attendances, r_choice)
                        agent_reward[agent, week, night] = get_reward(agent_attendances, r_choice)
                    else:
                        agent_propensity[agent] = propensity_update(agent_propensity[agent], np.where(agent_choice[agent,week,:]==1)[0][0], attendance, r_choice)
                        agent_reward[agent, week, night] = get_reward(attendance, r_choice)
            elfarol[week] = get_reward(agent_attendances, 1)

        # print([(elfarol.xk > elfarol.b).sum(), (elfarol.xk <= elfarol.b).sum()], ['Attendance > b', 'Attendance <= b'])
        # axs.plot(list(range(elfarol.total_weeks)), elfarol.global_reward, label="{0}".format(choice))

        if r_choice == 0:
            name = "local"
        elif r_choice == 1:
            name = "global"
        else:
            name = "difference"

        print("average {0} reward : {1}".format(name, elfarol.mean()))
        moving_avg = moving_average(elfarol, w)
        yerr = error_bars(elfarol, w)

        axs.errorbar(list(range(len(elfarol)))[0::10], moving_avg[0::10], yerr=yerr[0::10], label="{0}".format(name), marker=markers[choice], capsize=3)  # , uplims=True, lolims=True)#alpha=0.1)
        axs.legend()
        df['actions'] = list(range(k))
        df["reward"] = [name]*k
        df["propensities"] = (agent_propensity[22]/agent_propensity[22].sum()).ravel()
        df_list.append(df)
        # reset
        # sns.barplot(data=df, x="actions", y="propensities", ax=ax1[r_choice])
        # print(agent_propensity[:10]/agent_propensity[:10].sum())

        agent_propensity = np.zeros((n_agents, k))
        agent_choice = np.zeros((n_agents, n_weeks, k))
        agent_reward = np.zeros((n_agents, n_weeks, k))
        elfarol = np.zeros((n_weeks))
        temperature = 0
        df_tmp = pd.DataFrame.from_dict(day_wise_attendance_week, orient='index')
        print(df_tmp.mean(axis=1))
    df_t = pd.concat(df_list)
    # for i in range(3):
    sns.catplot(data=df_t, x="actions", y="propensities", kind="bar", ax=ax1, hue="reward")
    # ax1.set_xlabel("Actions")
    # ax1.set_ylabel("Propensities")
    # ax1.title("n="+str(n_agents)+ "k="+str(k)+ "b="+str(b))

    plt.show()