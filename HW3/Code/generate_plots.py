import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import  seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5) #, rc={"lines.linewidth": 2.5})
name = 'HW3'

import pandas as pd

with open('saved_files/n20b5k7.pkl', 'rb') as f:
    data = pickle.load(f)

agent_choice = data[0]
agent_reward = data[1]
xk = data[2] #attendance
# values in xk columns for dataframe
# df = pd.DataFrame(xk, columns=["S", "M", "T", "W", "R", "F", "St"][:-1])
df = pd.DataFrame(xk, columns=["S", "M", "T", "W", "R", "F", "St"])
df1 = pd.DataFrame()
# days = ["S", "M", "T", "W", "R", "F", "St"][:-1]
days = ["S", "M", "T", "W", "R", "F", "St"]
days_new=[[days[i]]*len(df[days[i]]) for i in range(len(days))]

df1["day"] = list(np.concatenate(days_new))
attendance = [df[days[i]] for i in range(len(days))]
attendance_ = list(np.concatenate(attendance))
df1["attendance"] = attendance_
#barplot of agent 2 attendances of each day##
nights_agent_attended = agent_choice[:,:,2].sum(axis=0)
## histogram of attendance##
def histogram(df, x, y, title):
    sns.histplot(df, x=x, y=y, bins=20, kde=True)
    plt.ylabel("frequency")
    plt.title(title)
    plt.show()

# plt.show()
#### Box plot ####
def boxPlot(df, x, y):
    g = sns.boxplot(x=x, y=y, data=df)
    sns.despine()
    plt.axhline(y=4, linewidth=0.5, linestyle="-", color="orange", clip_on=True)
    sns.set_axis_labels("Days", "Attendance")
    plt.show()
    g.legend.set_title("")


# g = sns.FacetGrid(data=df1, row="day", hue="day", height=1.5, aspect=5).map(sns.kdeplot, "attendance", clip_on=False, fill=True).add_legend()

### Ridge plot #####
def ridgePlot(df, x, y):
    g = sns.FacetGrid(data=df, row=x, hue=x, height=1.5, aspect=5).map(sns.kdeplot, y, clip_on=False, fill=True).add_legend()
    sns.despine()
    plt.axhline(y=4, linewidth=0.5, linestyle="-", color="orange", clip_on=True)
    sns.set_axis_labels("Days", "Attendance")
    g.set_titles(col_template="", row_template="")
    plt.title()
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.show()
    g.legend.set_title("")

def ridgeWithHistogram(df, x, y):
    g = sns.FacetGrid(data=df, row=x, hue=x, height=1.5, aspect=5).map(sns.histplot, y, clip_on=False, kde=True).add_legend()
    g.map(plt.axvline, x=4, linewidth=0.5, linestyle="-", color="red", clip_on=True)
    plt.axhline(y=4, linewidth=0.5, linestyle="-", color="orange", clip_on=True)
    sns.set_axis_labels("Days", "Attendance")
    g.set_titles(col_template="", row_template="")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.map(sns.histplot, y, bins=20, kde=True)
    g.legend.set_title("")
    sns.despine()
    plt.show()

if __name__ == "__main__":
    #arg parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--histogram', action='store_true', help='histogram of attendance')
    parser.add_argument('--boxplot', action='store_true', help='boxplot of attendance')
    parser.add_argument('--ridgeplot', action='store_true', help='ridgeplot of attendance')
    parser.add_argument('--ridgehist', action='store_true', help='ridgeplot with histogram of attendance')
    args = parser.parse_args()
    if args.histogram:
        histogram(df1, "attendance", "day", "Histogram of attendance")
    if args.boxplot:
        boxPlot(df1, "day", "attendance")
    if args.ridgeplot:
        ridgePlot(df1, "day", "attendance")
    if args.ridgehist:
        ridgeWithHistogram(df1, "day", "attendance")

# # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# # g.figure.subplots_adjust(hspace=-0.25)
#
# g.set_titles("")
# g.set_titles(col_template="", row_template="")
# plt.title()
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)

# g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# g.map(plt.axvline, x=5, linewidth=0.5, linestyle="-", color="red", clip_on=True)

## Ridge with histogram ##
# g = sns.FacetGrid(data=df1, row="day", hue="day", height=1.5, aspect=5).map(sns.histplot, "attendance", clip_on=False, kde=True).add_legend()
# g.map(plt.axvline, x=4, linewidth=0.5, linestyle="-", color="red", clip_on=True)
# plt.savefig("ridge_hist_plot_n50b4k6.pdf")
# plt.show()
########################
# g = sns.barplot(x=days, y=nights_agent_attended)
# g = sns.histplot(data=df1, x="attendance", hue="day", multiple="stack", bins=20)
# g = sns.boxplot(x="day", y="attendance", data=df1)
# Load data
# fig, axs = plt.subplots(1, 1)
# axs.set_xlabel('Week')
# axs.set_ylabel('System Reward')
# marker = ['o', 's', 'D', 'v', 'p', 'h', '8', 'P', 'X', 'd', 'H', 'x', 'D', 'p', 's', 'o', 'v', '8', 'P', 'H', 'x']
# for i in range(2):
#     if i==0:
#         name = "exponential local reward"
#         with open('exp_local_reward.pkl', 'rb') as f:
#             data = pickle.load(f)
#     else:
#         name = "hyperbolic local reward"
#         with open('lin_local_reward.pkl', 'rb') as f:
#             data = pickle.load(f)
#     moving_avg_global_reward = data[0]
#     yerr = data[1]
#     axs.errorbar(list(range(len(moving_avg_global_reward)))[0::10], moving_avg_global_reward[0::10], yerr=yerr[0::10], label="{0}".format(name), marker=marker[i], capsize=3)
# axs.legend()
# plt.savefig('local_reward.pdf')
# plt.show()
