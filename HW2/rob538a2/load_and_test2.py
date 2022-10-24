# from my_dqn_img import *
import random

from dqn_p2_state import *
# from dqn_p3_state import *
import cv2


def plot_image_overlay(imgs):
    img1 = imgs[0]
    for i in range(1, len(imgs)):
        img1 = cv2.addWeighted(img1, 0.9, imgs[i] , 0.1, -1, img1)
        # img1 = overlay_two_image_v2(img1, imgs[i], ignore_color=[255,0,0,])
    hsvImg = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    # multiple by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * 5
    # convert the HSV image back to RGB format
    img1 = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    plt.imshow(np.array(img1, dtype=np.uint8))
    plt.show()
    return img1

dqn = DQN(4)
dqn.eval_net_1.load_state_dict(torch.load('dqn_2agents_p3_a1.pth'))
dqn.eval_net_2.load_state_dict(torch.load('dqn_2agents_p3_a2.pth'))
#
# dqn.eval_net_1.load_state_dict(torch.load('dqn_2agents_p2_a1.pth'))
# dqn.eval_net_2.load_state_dict(torch.load('dqn_2agents_p2_a2.pth'))


# time.sleep(0.5)
steps = []
# for i in tqdm(range(100)):
pos1 = [(0, 3), (3, 5), (0, 0), (3, 7), (2, 2), (2, 5), (1, 4), (3, 7), (3, 4), (1, 4), (2, 0), (2, 3), (3, 1), (3, 8), (0, 7), (2, 7), (3, 5), (3, 2), (3, 2), (0, 8), (0, 8), (0, 3), (0, 4), (1, 8), (3, 7), (2, 8), (0, 0), (0, 2), (0, 7), (2, 7), (2, 0), (1, 1), (2, 1), (2, 7), (3, 0), (1, 2), (0, 4), (2, 7), (1, 1), (0, 0), (1, 3), (0, 2), (2, 6), (3, 4), (2, 3), (2, 2), (1, 8), (2, 3), (2, 1), (2, 2), (3, 1), (2, 6), (3, 6), (0, 6), (0, 7), (3, 3), (0, 7), (3, 7), (2, 0), (0, 4), (2, 1), (3, 8), (2, 5), (0, 5), (1, 0), (3, 5), (1, 2), (0, 8), (0, 3), (3, 2), (0, 3), (1, 8), (3, 5), (0, 1), (1, 7), (3, 3), (3, 3), (3, 8), (0, 1), (1, 3), (3, 7), (3, 3), (2, 2), (1, 1), (1, 8), (2, 4), (3, 6), (3, 5), (3, 0), (0, 1), (1, 4), (0, 6), (3, 6), (0, 8), (0, 8), (2, 4), (2, 5), (1, 7), (2, 8), (3, 4)]
pos2 = [(3, 2), (1, 3), (1, 5), (0, 5), (0, 0), (3, 6), (3, 6), (3, 8), (0, 8), (1, 6), (1, 1), (3, 4), (0, 1), (2, 4), (2, 0), (0, 5), (1, 1), (3, 0), (1, 1), (1, 0), (0, 3), (2, 4), (3, 8), (3, 8), (0, 3), (0, 2), (0, 6), (0, 3), (0, 8), (2, 0), (2, 5), (0, 0), (0, 2), (0, 0), (2, 2), (0, 7), (3, 6), (0, 8), (0, 0), (2, 5), (3, 7), (3, 5), (1, 5), (0, 2), (3, 2), (3, 3), (0, 3), (2, 6), (1, 8), (3, 0), (2, 0), (3, 5), (2, 2), (2, 5), (0, 5), (2, 8), (1, 0), (0, 7), (2, 6), (1, 4), (2, 3), (3, 1), (1, 2), (1, 0), (0, 0), (3, 1), (2, 6), (0, 4), (2, 2), (0, 3), (0, 5), (3, 0), (3, 5), (3, 0), (0, 7), (1, 1), (0, 8), (3, 3), (0, 1), (1, 1), (2, 6), (0, 6), (2, 1), (0, 5), (0, 3), (1, 4), (0, 1), (0, 3), (1, 7), (0, 6), (1, 0), (0, 6), (1, 0), (3, 5), (1, 0), (2, 5), (1, 1), (1, 1), (1, 0), (0, 7)]
rewards1 = []
for i in range(100):
    agent_pos_1 = pos1[i]
    agent_pos_2 = pos2[i]
    done = False

    # agent_pos_1 = (random.randint(0,3), random.randint(0,8))
    # agent_pos_2 = (random.randint(0,3), random.randint(0,8))

    state_1 = list(agent_pos_1) + list(agent_pos_2)
    state_1 = torch.FloatTensor(state_1)

    state_2 = list(agent_pos_1) + list(agent_pos_2)
    state_2 = torch.FloatTensor(state_2)
    grid, goals = custom_grid(2)

    # render([agent_pos_1, agent_pos_2], [0,0], grid)
    # time.sleep(2)
    # imgs = []
    t_rew = 0
    total_steps = 0
    while True:
        total_steps += 1
        action_1 = dqn.choose_action(state_1, dqn.eval_net_1, 0.5)
        action_2 = dqn.choose_action(state_2, dqn.eval_net_2, 0.5)

        agent_positions, rewards, done, goals, neighbours, img = step(grid, [action_1, action_2],[agent_pos_1, agent_pos_2], goals)
        # imgs.append(img)
        next_state_1 = list(agent_positions[0]) + list(agent_positions[1])
        next_state_2 = list(agent_positions[0]) + list(agent_positions[1])
        next_state_1 = torch.FloatTensor(next_state_1)
        next_state_2 = torch.FloatTensor(next_state_2)
        agent_pos_1 = agent_positions[0]
        agent_pos_2 = agent_positions[1]
        state_1 = next_state_1
        state_2 = next_state_2
        t_rew += rewards[0]+rewards[1]
        # print('action: ', action)
        # print('agent_pos: ', agent_pos)
        # print('reward: ', reward)
        # print('done: ', done)
        # print('goals: ', goals)
        # print('neighbours: ', neighbours)
        # print('next_state: ', next_state)
        # render(agent_positions, [action_1, action_2], grid)
        # time.sleep(0.5)

        if done:
            steps.append(total_steps)
            rewards1.append(t_rew)
            # plot_image_overlay(imgs)
            break




print(steps)
print(np.array(steps).mean())
print(rewards1)
print(np.array(rewards1).mean())