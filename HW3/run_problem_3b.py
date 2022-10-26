from elfarol_hw3 import *

def run_problem_3b():
    n_agents = 50
    b = 4
    k = 6
    elfarol = ElFarol(n_agents=n_agents, b=b, k=k)
    # choice 1 is  the night reward
    # choice 2 is the difference reward
    fig, axs = plt.subplots(1, 1)
    # fig1, axs1 = plt.subplots(1, 1)
    # fig2, axs2 = plt.subplots(1, 1)
    axs.set_xlabel('Week')
    axs.set_ylabel('System Reward')
    axs.set_title("n="+str(n_agents)+ "k="+str(k)+ "b="+str(b))

    # axs1.set_xlabel('Week')
    # axs1.set_ylabel('Local Reward')
    w=100
    markers = ['o', 'v', 's', 'p', 'P', '*', 'X', 'D', 'd', '1', '2', '3', '4', 'h', 'H', '+', 'x', 'X', 'D', 'd', '1', '2', '3', '4', 'h', 'H', '+', 'x']
    choices = [0, 1, 2, 3]
    agent_choice_ = defaultdict(list)
    agent_reward_ = defaultdict(list)
    elfarol_ = defaultdict(list)
    for choice in [0, 1, 2, 3]:
        agent_choice, agent_reward = elfarol.q_learning(choice=choice)
        agent_choice_[choice] = agent_choice
        agent_reward_[choice] = agent_reward
        elfarol_[choice] = [elfarol.global_reward,elfarol.xk]

        # factordness, sensitivity = factordness_and_sensitivity(elfarol, agent_i=2, choice=choice)
        # factordness = np.array(factordness)
        # print("factordness ", np.sum(factordness==-1), np.sum(factordness==1), "choice ", choice)
        # sensitivity = np.array(sensitivity)
        # print("sensitivity ", np.sum(sensitivity/elfarol.total_weeks))
        # night_rewards_agents = agent_choice * agent_reward
        # axs.plot(night_rewards_agents[:, :, 1][night_rewards_agents[:, :, 1] != 0], label='agent 1 LR')
        #
        # nights_agent_attended = elfarol.xk[agent_choice[:,:,2]==1]
        # factordness_agent_i = factordness[agent_choice[:, :, 2].ravel() == 1]
        # attendance_on_nights_agent_i_attended = elfarol.xk.ravel()[agent_choice[:, :, 2].ravel() == 1]
        # df = pd.DataFrame({'factordness': factordness_agent_i, 'attendance': attendance_on_nights_agent_i_attended , 'nights':list(range(len(nights_agent_attended)))})
        # sns.relplot(x="nights", y="attendance", size="factordness", data=df, ax=axs2)
        # agent3_rewards = night_rewards_agents[:,:, 3]
        # agent9_rewards = night_rewards_agents[:, :, 9]
        # agent15_rewards = night_rewards_agents[:, :, 15]
        # agent3_rewards=moving_average(agent3_rewards.sum(axis=1), w)

        name = str
        if choice == 0:
            name = 'global reward'
        if choice==1:
            name = "local rewards"
        if choice==2:
            name = "difference reward"
        if choice==3:
            name = "local difference reward"

        # axs1.plot(list(range(len(agent3_rewards))), agent3_rewards, label="agent 3 {0}".format(name))
        # axs1.plot(list(range(elfarol.total_weeks)), agent9_rewards[agent9_rewards!=0], label="agent 9 {0}".format(choice))
        # axs1.plot(list(range(elfarol.total_weeks)), agent15_rewards[agent15_rewards!=0], label="agent 15 {0}".format(choice))
        # agent1rewards = night_rewards_agents[:, :, 1]
        # axs.plot(list(range(elfarol.total_weeks)), agent1rewards[agent1rewards!=0])
        moving_avg_global_reward = moving_average(elfarol.global_reward, w)
        print("choice {0}, average_global_reward {1}".format(name, np.array(elfarol.global_reward).mean()))
        agent_attendances = get_attendance_nights(elfarol, agent_choice)
        # print(agent_reward)
        # agent_reward[:, :, 1].sum(axis=1)
        attendance_per_night = agent_choice[:, :, :].sum(axis=2)
        print("nash equilibrium ", attendance_per_night.mean(axis=0))
        print([(elfarol.xk > elfarol.b).sum(), (elfarol.xk <= elfarol.b).sum()],['Attendance > b', 'Attendance <= b'])
        # axs.plot(list(range(elfarol.total_weeks)), elfarol.global_reward, label="{0}".format(choice))
        yerr = error_bars(elfarol.global_reward, w)

        axs.errorbar(list(range(len(moving_avg_global_reward)))[0::10], moving_avg_global_reward[0::10], yerr=yerr[0::10], label="{0}".format(name), marker=markers[choice], capsize=3)#, uplims=True, lolims=True)#alpha=0.1)
        # axs.plot(list(range(len(moving_avg_global_reward))), moving_avg_global_reward)
        elfarol.reset()

    # max_weekly_reward = [elfarol.b*elfarol.k*np.exp(-1)]*elfarol.total_weeks
    # axs.plot(list(range(elfarol.total_weeks)), max_weekly_reward, color='green', linestyle='dashed')
    axs.legend()
    # axs1.legend()
    # axs[2].plot(list(range(elfarol.total_weeks*elfarol.k)), elfarol.xk.ravel())
    filename = "alldata_n"+str(n_agents)+"b"+str(b)+"k"+str(k)+".pkl"
    with open(filename, 'wb') as f:
        pickle.dump([agent_choice_, agent_reward_, elfarol_], f)
    plt.show()

if __name__ == '__main__':
    run_problem_3b()

