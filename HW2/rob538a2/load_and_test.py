from my_dqn_img import *
from my_grid import *
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
dqn.eval_net.load_state_dict(torch.load('dqn.pth'))
done = False
grid, goals = custom_grid(1)

agent_pos = [(2, 8)]
state = render(agent_pos, [0], grid)
state = torch.FloatTensor(state)
state = state.permute(2, 0, 1)
imgs = []
while True:
    action = [dqn.choose_action(state, 0)]
    agent_pos, reward, done, goals, neighbours, next_state = step(grid, action, agent_pos, goals)
    imgs.append(next_state)
    next_state = torch.FloatTensor(next_state)
    next_state = next_state.permute(2, 0, 1)
    state = next_state
    print('action: ', action)
    print('agent_pos: ', agent_pos)
    print('reward: ', reward)
    print('done: ', done)
    print('goals: ', goals)
    # print('neighbours: ', neighbours)
    # print('next_state: ', next_state)
    # render(agent_pos, action, grid)
    # time.sleep(0.5)
    if done:
        plot_image_overlay(imgs)
        break