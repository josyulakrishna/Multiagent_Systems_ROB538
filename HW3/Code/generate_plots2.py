import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5) #, rc={"lines.linewidth": 2.5})

#read file and get data
with open('saved_files/alldata_n20b5k7.pkl', 'rb') as f:
    data = pickle.load(f)

agent_choice = data[0]
agent_reward = data[1]
r_xk = data[2] #attendance

df = pd.DataFrame()
days = ["S", "M", "T", "W", "R", "F", "St"]

def get_attendance_nights(agent_choice):
    agent_attendance_nights = []
    for agent_i in range(agent_choice.shape[2]):
        agent_attendance_nights.append(agent_choice[:,:,agent_i].sum(axis=0))
    return agent_attendance_nights
dfs = []
for choice in range(4):
    agent_choice_ = agent_choice[choice]
    agent_reward_ = agent_reward[choice]
    global_reward = r_xk[choice][0]
    xk = r_xk[choice][1]
    if choice == 0:
        name = 'global_reward'
    if choice == 1:
        name = "local_reward"
    if choice == 2:
        name = "difference_reward"
    if choice == 3:
        name = "local_difference_reward"
    # print(get_attendance_nights(agent_choice_)[2])
    df1 = pd.DataFrame() #[get_attendance_nights(agent_choice_)[2]], columns=["S", "M", "T", "W", "R", "F"])
    df1["count"]=get_attendance_nights(agent_choice_)[2]
    df1["day"]=days
    df1["reward"]=[name]*len(days)
    dfs.append(df1)
# print(pd.concat(dfs))
dfs = pd.concat(dfs, ignore_index=True)
# print(dfs)
sns.factorplot(data=dfs, x="day", hue="reward", y="count", kind="bar")
plt.savefig("attendance_nights.pdf")
plt.show()
