import math

import torch
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from torch import nn
from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np


# This multi-agent controller shares parameters between agents
class HDQRlCardMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(HDQRlCardMAC, self).__init__(scheme, groups, args)
        self.args = args
        self.inputs_alpha = None
        e = self.input_shape
        self.fc1 = nn.Sequential(
            nn.Linear(e+54 + 1, 2 * e),
            nn.ReLU(inplace=True),
            nn.Linear(2 * e, e+54),
            nn.Sigmoid()
        ).to('cuda')
        self.fc2 = nn.Sequential(
            nn.Linear(e+54, 2 * e),
            nn.ReLU(inplace=True),
            nn.Linear(2 * e, e+54),
            nn.Sigmoid()
        ).to('cuda')
        self.mu = args.mu
        self.theta = args.theta
        self.agent_rule = DouDizhuRuleAgentV1()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = [ep[t_ep] for ep in ep_batch["avail_actions"]]
        player_id = th.tensor([ep[t_ep] for ep in ep_batch["player_id"]])
        if all(player_id[bs]):
            actions = []
            for i in bs:
                action = self.agent_rule.step(ep_batch["obs"][i][t_ep])
                action_index = list(ep_batch["obs"][i][t_ep]["legal_actions"].keys())[ep_batch["obs"][i][t_ep]["raw_legal_actions"].index(action)]
                actions.append(action_index)
            return actions
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions = self.action_selector.select_action([qvals[i] for i in range(len(bs))], t_env, test_mode=test_mode)



        chosen_actions = chosen_actions.tolist()
        if isinstance(chosen_actions, (int, np.integer)):
            chosen_actions = [chosen_actions]
        # 获取所有动作 key
        action_keys_list = [list(avail_actions[i].keys()) for i in bs]

        # 选中对应 key
        selected_keys = [action_keys_list[i][chosen_actions[i]] for i in range(len(chosen_actions))]
        return selected_keys

    def forward(self, ep_batch, t, test_mode=False, alpha_q=False):
        agent_inputs, batch_action_counts, bs = self._build_inputs(ep_batch, t)
        player_id = th.tensor([ep[t] for ep in ep_batch["player_id"]], device=agent_inputs.device)
        if test_mode:
            self.agent.eval()

        hidden_states = th.gather(
            self.hidden_states,
            dim=1,
            index=player_id.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_states.size(2))
        ).squeeze(1)

        batch_action_counts_tensor = th.tensor(batch_action_counts, device=hidden_states.device)
        hidden_state_expanded = hidden_states[bs].repeat_interleave(batch_action_counts_tensor, dim=0)

        if t == 0:
            self.inputs_alpha =torch.rand_like(agent_inputs, device=agent_inputs.device)

        inputs_alpha= self.fc2(self.inputs_alpha)

        agent_inputs_temp = agent_inputs

        agent_inputs = agent_inputs_temp*math.cosh(self.theta)
        inputs_alpha = agent_inputs_temp*math.sinh(self.theta)+ math.exp(-self.theta)*inputs_alpha

        agent_outs, hidden_state_expanded = self.agent(agent_inputs, hidden_state_expanded)

        bad_output, _ = self.agent(inputs_alpha, hidden_state_expanded)
        agent_outs = th.split(agent_outs, batch_action_counts)
        bad_output = th.split(bad_output, batch_action_counts)
        agent_outs = tuple(a - b for a, b in zip(agent_outs, bad_output))
        # 将 agent_outs 最后一维改成最大值的位置
        max_pos = torch.argmax(agent_outs, dim=-1)  # 形状变为 (batch, n)
        # 加1后扩展回原来的维度结构
        max_pos = max_pos.unsqueeze(-1)
        bad_input = self.fc1(torch.cat([inputs_alpha, max_pos], dim=-1))  # 形状 (b, a, e)
        self.inputs_alpha = ((1-self.mu) * self.inputs_alpha + self.mu * bad_input)



        return agent_outs

    def _build_inputs(self, batch, t):
        avail_actions = [ep[t] for ep in batch["avail_actions"]]
        obs = [ep[t]["obs"] if ep[t].get("obs") is not None else np.zeros(256) for ep in batch["obs"]]
        valid_indices = [i for i, acts in enumerate(avail_actions) if acts]
        bs = batch.batch_size
        device = 'cuda' if th.cuda.is_available() else 'cpu'

        expanded_obs = []
        action_vectors = []
        batch_action_counts = []

        for b in range(bs):
            cur_obs = obs[b]
            actions = avail_actions[b]
            n_actions = len(actions)

            expanded_obs.append(np.tile(cur_obs, (n_actions, 1)))

            vecs = [np.array(vec, dtype=np.float32) for vec in actions.values()]
            if vecs == []:
                continue
            action_vectors.append(np.stack(vecs))  # (n_actions, action_dim)
            batch_action_counts.append(n_actions)

        # 将列表合并为 tensor
        expanded_obs_tensor = th.tensor(np.concatenate(expanded_obs, axis=0), dtype=th.float32,
                                        device=device)  # (total_n_actions, 250)
        action_vec_tensor = th.tensor(np.concatenate(action_vectors, axis=0), dtype=th.float32,
                                      device=device)  # (total_n_actions, action_dim)

        # 拼接成 input: (total_n_actions, 250 + action_dim)
        inputs = th.cat([expanded_obs_tensor, action_vec_tensor], dim=1)  # (total_n_actions, 250 + action_dim)

        return inputs, batch_action_counts, valid_indices

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        # if self.args.obs_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape += self.n_agents

        return input_shape
