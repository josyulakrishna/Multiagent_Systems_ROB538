import random
import time
from grid import *
from rendering import *
from window import *

window = Window('my_custom_env')
window.show(block=False)



def transition(self, row: int, col: int, a: int, goals: int) -> tuple[int, float, bool, int]:
    """
    Compute next state, reward and done when starting at (row, col)
    and taking the action action a.
    """
    newrow, newcol = self.to_next_xy(row, col, a)
    # newstate = self.to_s(newrow, newcol)
    newletter = self.desc[newrow, newcol]
    newstate = (newrow, newcol)
    done = False
    if bytes(newletter) in b"GW":
        self.desc[newrow, newcol] = b'E'
        self.set(newrow, newcol, None)
        goals -= 1
    if goals <= 0:
        done = True
    reward = self.reward_map[newletter]
    return newstate, reward, done, goals


# goals = 2
# agent_pos = [(0, 2), (0, 1)]

def getNeighbours(row, col, desc):
        neighbours = []
        if row > 0:
            letter = desc[row - 1, col]
            if letter == b'G':
                neighbours.append(1)
            else:
                neighbours.append(0)
        if row < len(desc) - 1:
            letter = desc[row + 1, col]
            if letter == b'G':
                neighbours.append(1)
            else:
                neighbours.append(0)
        if col > 0:
            letter = desc[row, col - 1]
            if letter == b'G':
                neighbours.append(1)
            else:
                neighbours.append(0)
        if col < len(desc[1]) - 1:
            letter = desc[row, col + 1]
            if letter == b'G':
                neighbours.append(1)
            else:
                neighbours.append(0)
        return neighbours

def render(agent_pos, agent_dir, grid):
    img = grid.render(
        tile_size=32,
        agent_pos=[*agent_pos],
        agent_dir=agent_dir,
    )
    # res = cv2.resize(img, dsize=(210, 160), interpolation=cv2.INTER_CUBIC)
    window.show_img(img)
    return img

def step(grid, actions, agent_positions, goals):
    r = [None]* len(agent_positions)
    neighbours = []
    for i in range(len(actions)):
        agent_positions[i], r[i], d, goals = transition(grid, agent_positions[i][0], agent_positions[i][1], actions[i], goals)
        neighbours += getNeighbours(agent_positions[i][0], agent_positions[i][1], grid.desc)
    if len(neighbours) < 8:
        neighbours += [0] * (8 - len(neighbours))
    img = grid.render(
        tile_size=32,
        agent_pos=[*agent_positions],
        agent_dir=actions,
    )
    return agent_positions, r, d, goals, neighbours, img


def custom_grid(agents):


    w, h = 5, 10
    desc = []
    if agents == 2:
        goals = 2
        desc = np.array([[b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'G'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'G', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E']], dtype='|S1')
    if agents == 1:
        goals = 1
        # for agent its row col, for goal its col row
        # ex: agent(0,1) places agent at col 0, row 1, (row is top to bottom, col is left to right)
        # goal(1,0) places goal at row 1, col 0, (row is left to right, col is top to bottom)
        desc = np.array([[b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'G', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E'],
                         [b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E', b'E']], dtype='|S1')

    reward_map = {
        b'E': -1.0,
        b'S': 0.0,
        b'W': 0.0,
        b'G': 20.0,
    }

    grid = SimpleGrid(width=w, height=h, desc=desc, reward_map=reward_map)

    for row in range(w):
        for col in range(h):
            letter = desc[row, col]
            if letter == b'G':
                grid.set(row, col, Goal())
            elif letter == b'W':
                grid.set(row, col, Wall(color='black'))
            else:
                grid.set(row, col, None)

    return grid, goals
#
# goals = 1
# agent_pos = [(0, 0)]
#
# grid, goals = custom_grid(1)
# render(agent_pos, [0], grid)
# # time.sleep(3)
# for k in range(100):
#     actions = [random.randint(0, 4) for i in range(len(agent_pos))]
#     agent_pos, r, d, goals, neighbours, img = step(grid, actions, agent_pos, goals)
#     print(agent_pos, r, d, goals, neighbours, img.shape)
#     render(agent_pos, actions, grid)
#     time.sleep(0.5)
