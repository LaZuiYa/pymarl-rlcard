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
class RlCardMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(RlCardMAC, self).__init__(scheme, groups, args)
        self.args = args
        self.inputs_alpha = None
        e = self.input_shape
        self.model_player_id = []
        self.agent_rule = DouDizhuRuleAgentV1()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = [ep[t_ep] for ep in ep_batch["avail_actions"]]
        player_id = th.tensor([ep[t_ep] for ep in ep_batch["player_id"]])[bs]
        if len(player_id) == 0:
            player_id = [-1]
        if (player_id[0] not in self.model_player_id) or bs == []:
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
        # player_id = th.tensor([ep[t] for ep in ep_batch["player_id"]], device=agent_inputs.device)
        if test_mode:
            self.agent.eval()


        agent_outs = self.agent(agent_inputs)
        agent_outs = th.split(agent_outs, batch_action_counts)


        return agent_outs

    def _build_inputs(self, batch, t):
        avail_actions = [ep[t] for ep in batch["avail_actions"]]
        obs = [ep[t]["obs"] if ep[t].get("obs") is not None else np.zeros(259) for ep in batch["obs"]]
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
        if len(action_vectors) == 0:
            action_vec_tensor = th.zeros((0, 0), dtype=th.float32, device=device)
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
