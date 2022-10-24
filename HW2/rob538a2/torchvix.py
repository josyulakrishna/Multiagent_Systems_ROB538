from torchviz import make_dot
# from my_dqn_img import *
from dqn_p2_state import *

dqn = DQN(4)
# state_1 = torch.randn(1, 3, 320,160, dtype=torch.float).cuda()
# make_dot(dqn.eval_net(state_1), params=dict(dqn.eval_net.named_parameters())).render("rnn_torchviz", format="png")
print(dqn.eval_net_1)