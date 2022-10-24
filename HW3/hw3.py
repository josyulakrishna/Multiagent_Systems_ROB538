import numpy as np
import itertools
import matplotlib
from collections import defaultdict
import matplotlib.pyplot as plt

#El Farol bar problem
# z_prime is the state where action of agent i is different from the system state
# state of agent is 0, 1
# system state is a collection of agent attendance  to bar.
# x_k is the attendance on night k
# b is the optimal number of people in the bar
# global reward is the system reward per week


class ElFarol :
    def __init__(self, n_agents = 20, b = 5, k= 6):
        self.n_agents = n_agents
        self.b = b
        self.state = np.zeros(n_agents)
        # self.x = None #attendance on night
        self.iterations = 10000
        self.k = k
        self.gamma = 0.99
        self.total_weeks = 600
        self.global_reward = []
        self.xk = np.zeros((self.total_weeks, self.k)) #collection of attendance over k nights
        self.z = np.zeros(( self.total_weeks, k, n_agents)) #state of agent i
        self.agent_estimate = np.zeros((n_agents, k))
        self.agent_buffer = [[] for i in range(n_agents)]
        self.global_buffer = []
        self.night_rewards = np.zeros((self.total_weeks, n_agents, self.k))
        self.V = defaultdict(lambda: defaultdict(list))


    def chooseAction(self, agent_i, reward_choice=1):
        if reward_choice=="random":
            action = np.random.choice([0,1])
        if reward_choice=="greedy":
            action = np.argmax(self.agent_estimate[agent_i])
        if reward_choice=="epsilon":
            epsilon = 0.1
            if np.random.random() < epsilon:
                action = np.random.choice([0,1])
            else:
                action = np.argmax(self.agent_estimate[agent_i])
        return action

    def generate_fake_data(self):
        for week in range(self.total_weeks):
            for agent in range(self.n_agents):
                night = np.random.choice(self.k, 1)[0]
                self.z[week, night, agent] = 1
            for night in range(self.k):
                self.xk[week, night] = self.z[week, night, :].sum()

    def dynamics(self):
        for week in range(self.total_weeks):
            for night in range(self.k):
                attendance = []
                for agent_i in range(self.n_agents):
                    action = self.chooseAction(agent_i, reward_choice="random")
                    attendance.append(action)
                    self.z[week, night, agent_i ] = action
                self.xk[week, night ] = np.array(attendance).sum() #state of the bar on night k
            self.global_reward.append(self.week_reward(week))

    def moving_average(self):
        #get moving average reward of 3 previous nights
        rewards = []
        w = 3
        for i in range(len(self.buffer) - w + 1):
             rewards.append(sum(self.xk[i:i+w])/w)
        return rewards

    def insert_buffer(self, g, a, reward, agent_i):
        if g:
            if len(self.global_buffer) > 100:
                self.global_buffer.pop(0)
                self.global_buffer.append(reward)
            else:
                self.global_buffer.append(reward)

        if a:
            if len(self.agent_buffer[agent_i]) > 100:
                self.agent_buffer[agent_i].pop(0)
            else:
                self.agent_buffer[agent_i].append(reward)

    def week_reward(self, week):
        G = self.xk[week]@np.exp(-self.xk[week]/self.b)
        self.insert_buffer(g=True, a=False, reward=G, agent_i=None)
        return G

    def local_reward(self, reward_choice, x, week, agent_i):
        if reward_choice == 0:
            #constant reward
            gi = 10
        if reward_choice == 1:
            #weighted reward
            gi = x*np.exp(-x/self.b)
            self.insert_buffer(g=False, a=True, reward=gi, agent_i=agent_i)

        if reward_choice == 2:
            #percentage of global reward
            gi = 0.5*self.week_reward(week)
            self.insert_buffer(g=False, a=True, reward=gi, agent_i=agent_i)

        return gi

    def sensitivity(self, agent_i, k, week, reward_choice=1):
        """Return the sensitivity of gi in G."""
        # agent_i is the index of the agent
        # k is the number of nights
        # choice is the 0,1 attend or not
        z = self.z[week, k, :]

        # gi_z = self.local_reward(reward_choice, self.xk[week, k], week, agent_i)
        gi_z = np.exp(-z[k]/self.b)

        # self.night_rewards[week, agent_i, k] = gi_z

        z_i = z[k]

        z_prime = np.random.choice(self.n_agents, size = (self.k,))

        z_i_prime = z_prime[agent_i]

        # gi_z - gi(z-z_i+z_i_prime) z_tmp = z-z_i+z_i_prime
        #some gimmick to concatenate z and z_prime
        if agent_i== 0:
            z_tmp = np.concatenate(([z_i_prime], z[1:]))
        else:
            z_tmp = np.concatenate((z[:agent_i], [z_i_prime], z[agent_i+1:]))

        numerator = gi_z - self.local_reward(reward_choice, z_tmp.sum(), week, agent_i)

        if agent_i== 0:
            z_tmp = np.concatenate(([z_i], z_prime[1:]))
        else:
            z_tmp = np.concatenate((z_prime[:agent_i], [z_i], z_prime[agent_i+1:]))

        denominator = gi_z - self.local_reward(reward_choice, z_tmp.sum(), week, agent_i)

        return numerator/denominator

    def factordness(self, agent_i, k, week, reward_choice=1):
        # z_prime is a state which only differs from z in the state of component i
        z_i_prime = np.random.choice(self.n_agents, size = (1,))[0]
        z = self.xk[week, :]
        # gi_z = z[k]*np.exp(-z[k]/self.b)
        gi_z = np.exp(-z[k] / self.b)
        z[k] = z_i_prime
        # gi_z_prime = z[k]*np.exp(-z[k]/self.b)
        gi_z_prime = np.exp(-z[k]/self.b)
        G_z = self.global_reward[week]
        G_z_prime = z@np.exp(-z/self.b)

        if (gi_z-gi_z_prime)*(G_z-G_z_prime) > 0:
            return 1
        else:
            return -1

    ######### Learning Algorithms #########
    def q_learning(self):
        gamma = 0.99
        # delta = np.ones((self.n_agents, self.k))
        global_reward = []
        epsilon = 0.3
        agent_choice = np.zeros((self.total_weeks, self.k, self.n_agents))
        agent_reward = np.zeros((self.total_weeks, self.k, self.n_agents))

        for week in range(self.total_weeks):
            agent_attendances = np.zeros(self.k)
            for agent_i in range(self.n_agents):
                if np.random.rand() < epsilon:
                    night = np.random.choice(self.k, 1)[0]
                else:
                    night = np.argmax(self.agent_estimate[agent_i,:])
                agent_attendances[night] += 1
                agent_choice[week, night, agent_i] += 1
            for agent_i in range(self.n_agents):
                for night in range(self.k):
                    if agent_choice[week, night, agent_i] != 0:
                        reward = self.b / agent_attendances[night] if agent_attendances[night] < self.b else -1 * self.b /agent_attendances[night]
                        # reward = np.exp(-agent_attendances[night]/self.b) if agent_attendances[night] <= self.b else -1*np.exp(-agent_attendances[night]/self.b)
                        self.agent_estimate[agent_i, night] += (reward + gamma*np.max(self.agent_estimate[agent_i, :]) - self.agent_estimate[agent_i, night])

                        agent_reward[week, night, agent_i] = reward
                        self.agent_estimate[agent_i, night] = self.agent_estimate[agent_i, night] + 0.5*(reward + gamma*max(self.agent_estimate[agent_i, :]) - self.agent_estimate[agent_i, night])
            global_reward.append(agent_attendances@np.exp(-agent_attendances/self.b))
        return global_reward, agent_choice, agent_reward


    def fictious_play(self):
        #compute empirical distribution over crowded/uncrowded
        # a_t+1 = a_t + 1 if attendance[night]<= self.b else 0
        # p_t = a_t / t
        # reward = p_t*alpha_t + (1-p_t)*(1-alpha_t) if action =1 , else 0
        prob_uncrowded  = np.zeros(self.k,)
        agent_choice = np.zeros((self.total_weeks, self.k, self.n_agents))
        agent_reward = np.zeros((self.total_weeks, self.k, self.n_agents))
        global_reward = []
        total_nights = 0
        epsilon = 0.5
        for week in range(self.total_weeks):
            agent_attendances = np.zeros(self.k)
            for agent_i in range(self.n_agents):
                if np.random.rand() < epsilon:
                    night = np.random.choice(self.k, 1)[0]
                else:
                    night = np.argmax(self.agent_estimate[agent_i, :])

                agent_attendances[night] += 1
                agent_choice[week, night, agent_i] += 1
            total_nights += self.k
            prob_uncrowded[agent_attendances <= self.b] += 1
            prob_uncrowded[agent_attendances<=self.b] *= 1/total_nights
            global_reward.append(agent_attendances@np.exp(-agent_attendances/self.b))
            for agent_i in range(self.n_agents):
                for night in range(self.k):
                    if agent_choice[week, night, agent_i] != 0:
                        # alpha  = np.exp(-agent_attendances[night]/self.b)
                        alpha = self.b/agent_attendances[night]
                        self.agent_estimate[agent_i, night] = prob_uncrowded[night]*(alpha) - (1-prob_uncrowded[night])*(1-alpha)
                        agent_reward[week, night, agent_i] = self.agent_estimate[agent_i, night]
        return global_reward, agent_choice, agent_reward


    def policy_iteration(self):
        ## value iteration for MDP ##
        for agent in range(self.n_agents):
            for night in range(self.k):
                self.V[agent][night]=[0,0]
        theta = 0.01
        delta = np.ones((self.n_agents, self.k, 2))
        self.global_reward = []
        for week in range(self.iterations):
            agent_choice = defaultdict(int) #which night agent attends
            z = np.zeros(self.k) # total attendance of week per night
            for agent in range(self.n_agents):
                agent_choice[agent]=np.random.choice(self.k, size = (1,))[0]
                z[agent_choice[agent]] += 1
            self.global_reward.append(z@np.exp(-z/self.b))
            for agent in range(self.n_agents):
                for night in range(self.k):
                    reward = np.exp(-z[night] / self.b)
                    if night == agent_choice[agent]:
                        if delta[agent][night][1] > theta:
                            temp = self.V[agent][night][1]
                            V = max(self.V[agent][night][1], self.V[agent][night][0])
                            self.V[agent][night][1] = reward + self.gamma * V
                            delta[agent][night][1] = max(delta[agent][night][1], abs(temp - self.V[agent][night][1]))
                    else:
                        if delta[agent][night][0] > theta:
                            temp = self.V[agent][night][0]
                            V = max(self.V[agent][night][1], self.V[agent][night][0])
                            self.V[agent][night][0] = reward + self.gamma * V
                            delta[agent][night][0] = max(delta[agent][night][0], abs(temp - self.V[agent][night][0]))


    def select_night(self):
        agent_local_reward = defaultdict(list)
        global_reward = []
        agent_choice = np.zeros((self.total_weeks, self.k, self.n_agents))
        for week in range(self.total_weeks):
            for night in range(self.k):
                self.xk[week, night] = 0
                for agent in range(self.n_agents):
                    if self.V[agent][night][1] > self.V[agent][night][0]:
                        self.xk[week, night] += 1
                        agent_choice[week, night, agent] = 1
            for night in range(self.k):
                agent_local_reward[week].append(np.exp(-self.xk[week, night]/self.b))

            global_reward.append(self.xk[week, :]@np.exp(-self.xk[week, :]/self.b))
        return agent_local_reward, global_reward, agent_choice

    def q_select_night(self):
        global_reward = []
        for week in range(self.total_weeks):
            agent_attendances = np.zeros(self.k)
            for agent_i in range(self.n_agents):
                night = np.argmax(self.agent_estimate[agent_i, :])
                agent_attendances[night] += 1
            global_reward.append(agent_attendances@np.exp(-agent_attendances/self.b))
        return global_reward

def run_problem_1():
    #run experiment
    elfarol = ElFarol()
    # elfarol.dynamics()
    # factordness = []
    elfarol.generate_fake_data()

    for week in range(elfarol.total_weeks):
        elfarol.global_reward.append(elfarol.week_reward(week))

    agent_sensitivities = []
    agent_factordness = []

    for week in range(elfarol.total_weeks):
        for agent_i in range(elfarol.n_agents):
            for k in range(elfarol.k):
                if agent_i==3 and elfarol.z[week, k, agent_i] == 1:
                    # agent_sensitivities[elfarol.z[week, k, agent_i]].append(elfarol.factordness(agent_i, k, week))
                    agent_factordness.append(elfarol.factordness(agent_i, k, week))
                    agent_sensitivities.append(elfarol.sensitivity(agent_i, k, week))
    #plot global reward
    # plt.plot(list(range(elfarol.total_weeks)), elfarol.global_reward)
    # plt.plot(list(range(elfarol.total_weeks)), agent_factordness)
    # plt.plot(list(range(elfarol.total_weeks)), agent_sensitivities)
    # print(np.sum(np.array(agent_factordness)==-1),np.sum(np.array(agent_factordness)==1) )

    plt.show()
    return 0

def run_problem_2():
    elfarol = ElFarol(n_agents=20, b=5, k=7)
    elfarol.policy_iteration()
    # agent_reward, weekly_reward, agent_choice = elfarol.select_night()
    fig, axs = plt.subplots(1, 1)
    weekly_reward, agent_choice, agent_reward = elfarol.q_learning()
    # agent_reward_night = agent_choice@agent_reward
    # weekly_reward = elfarol.q_select_night()
    axs.plot(list(range(elfarol.total_weeks)), weekly_reward)
    # axs[1].plot(list(range(elfarol.total_weeks)), )
    # axs[2].plot(list(range(elfarol.total_weeks*elfarol.k)), elfarol.xk.ravel())
    plt.show()



if __name__ == "__main__":
    run_problem_2()
# 574450259 FRA