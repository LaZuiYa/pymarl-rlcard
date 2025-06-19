import math

import torch
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from torch import nn

from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class SDQ(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(SDQ, self).__init__(scheme, groups, args)
        self.args = args
        self.inputs_alpha = None
        e = self.input_shape
        self.fc1 = nn.Sequential(
            nn.Linear(e + 1, 2 * e),
            nn.ReLU(inplace=True),
            nn.Linear(2 * e, e),
            nn.Sigmoid()
        ).to('cuda')
        self.fc2 = nn.Sequential(
            nn.Linear(e, 2 * e),
            nn.ReLU(inplace=True),
            nn.Linear(2 * e, e),
            nn.Sigmoid()
        ).to('cuda')
        self.mu = args.mu
        self.theta = args.theta
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals  = self.forward(ep_batch, t_ep,test_mode=test_mode, alpha_q=True, avail_actions = avail_actions, t_env = t_env, bs=bs)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, alpha_q=False, avail_actions = None, t_env = None, bs=slice(None)):
        agent_inputs = self._build_inputs(ep_batch, t)

        if t == 0:
            self.inputs_alpha =torch.rand_like(agent_inputs, device=agent_inputs.device)

        inputs_alpha= self.fc2(self.inputs_alpha)

        agent_inputs_temp = agent_inputs
        #torch.clamp(agent_inputs, min=0.1,  max=1)
        # agent_inputs_temp = torch.where(agent_inputs == 0,
        #                            torch.tensor(-0.3, dtype=agent_inputs.dtype, device=agent_inputs.device),
        #                            agent_inputs)
        agent_inputs = agent_inputs_temp*math.cosh(self.theta)
        inputs_alpha = agent_inputs_temp*math.sinh(self.theta)+ math.exp(-self.theta)*inputs_alpha
        if test_mode:
            self.agent.eval()
        hidden_states = self.hidden_states
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        bad_output, _ = self.agent(inputs_alpha, hidden_states)
        agent_outs = agent_outs - bad_output
        # 将 agent_outs 最后一维改成最大值的位置
        max_pos = torch.argmax(agent_outs, dim=-1)  # 形状变为 (batch, n)
        # 加1后扩展回原来的维度结构
        max_pos = max_pos.unsqueeze(-1)
        bad_input = self.fc1(torch.cat([inputs_alpha, max_pos], dim=-1))  # 形状 (b, a, e)
        self.inputs_alpha = ((1-self.mu) * self.inputs_alpha + self.mu * bad_input)

        return agent_outs
    def parameters(self):
        #返回fc1,fc2,agent.parameters()
        params = list(self.fc1.parameters()) + list(self.fc2.parameters())  + list(self.agent.parameters())
        return params
# import math
#
# import torch
# from modules.agents import REGISTRY as agent_REGISTRY
# from components.action_selectors import REGISTRY as action_REGISTRY
# from torch import nn
#
# from .basic_controller import BasicMAC
# import torch as th
# from utils.rl_utils import RunningMeanStd
# import numpy as np
#
# # This multi-agent controller shares parameters between agents
# class SDQ(BasicMAC):
#     def __init__(self, scheme, groups, args):
#         super(SDQ, self).__init__(scheme, groups, args)
#         self.args = args
#         self.inputs_alpha = None
#         e = self.input_shape
#         self.fc1 = nn.Sequential(
#             nn.Linear(e + 1, 2 * e),
#             nn.ReLU(inplace=True),
#             nn.Linear(2 * e, e),
#             nn.Sigmoid()
#         ).to('cuda')
#         self.fc2 = nn.Sequential(
#             nn.Linear(e, 2 * e),
#             nn.ReLU(inplace=True),
#             nn.Linear(2 * e, e),
#             nn.Sigmoid()
#         ).to('cuda')
#         self.mu = args.mu
#         self.theta = args.theta
#     def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
#         # Only select actions for the selected batch elements in bs
#         avail_actions = ep_batch["avail_actions"][:, t_ep]
#
#
#         qvals  = self.forward(ep_batch, t_ep,test_mode=test_mode, alpha_q=True, avail_actions = avail_actions, t_env = t_env, bs=bs)
#
#         chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
#         max_pos = chosen_actions.unsqueeze(-1)
#         bad_input = self.fc1(torch.cat([self.ina[bs], max_pos], dim=-1))  # 形状 (b, a, e)
#
#         self.inputs_alpha[bs] = ((1-self.mu) * self.inputs_alpha[bs] + self.mu * bad_input)
#         return chosen_actions
#
#     def forward(self, ep_batch, t, test_mode=False, alpha_q=False, avail_actions = None, t_env = None, bs=slice(None)):
#         agent_inputs = self._build_inputs(ep_batch, t)
#
#         if t == 0:
#             self.inputs_alpha = torch.rand_like(agent_inputs, device=agent_inputs.device)
#
#         inputs_alpha= self.fc2(self.inputs_alpha)
#
#         agent_inputs_temp = agent_inputs #torch.clamp(agent_inputs, min=0.1,  max=1)
#         # agent_inputs_temp = torch.where(agent_inputs == 0,
#         #                            torch.tensor(-0.5, dtype=agent_inputs.dtype, device=agent_inputs.device),
#         #                            agent_inputs)
#         agent_inputs = agent_inputs_temp*math.cosh(self.theta)
#         inputs_alpha = agent_inputs_temp*math.sinh(self.theta)+ inputs_alpha #agent_inputs_temp*math.sinh(0.8) +inputs_alpha
#
#         if test_mode:
#             self.agent.eval()
#         hidden_states = self.hidden_states
#         agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
#         bad_output, _ = self.agent(inputs_alpha, hidden_states)
#
#         # 将 agent_outs 最后一维改成最大值的位置
#         # max_pos = torch.argmax(agent_outs, dim=-1)  # 形状变为 (batch, n)
#         # # 加1后扩展回原来的维度结构
#         # max_pos = (max_pos + 1).unsqueeze(-1)
#         self.ina = inputs_alpha
#         if avail_actions is None:
#             max_pos = torch.argmax(agent_outs, dim=-1)  # 形状变为 (batch, n)
#             # 加1后扩展回原来的维度结构
#             max_pos = max_pos.unsqueeze(-1)
#             bad_input = self.fc1(torch.cat([inputs_alpha, max_pos], dim=-1))
#             self.inputs_alpha = ((1 - self.mu) * self.inputs_alpha + self.mu * bad_input)
#         agent_outs = agent_outs - bad_output
#
#         return agent_outs
#     def parameters(self):
#         #返回fc1,fc2,agent.parameters()
#         params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.agent.parameters())
#         return params
