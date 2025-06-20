import copy

from ray.train.examples.pytorch.torch_data_prefetch_benchmark.auto_pipeline_for_host_to_device_data_transfer import \
    train_func
from rlcard.games.doudizhu.utils import action

from components.episode_buffer import EpisodeBatch
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from utils.rl_utils import build_rlcard_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num


class RLCardLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = mac.parameters()

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def build_util(self, tensor, n_agents=3):
        slices = [tensor[:, i::n_agents, ...] for i in range(n_agents)]
        return th.cat(slices, dim=0)
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        rewards = self.build_util(rewards, n_agents=self.n_agents)
        mask = self.build_util(mask, n_agents=self.n_agents)
        terminated = self.build_util(terminated, n_agents=self.n_agents)
        # rewards0 = rewards[:, ::3, :]
        # rewards1 = rewards[:, 1::3, :]
        # rewards2 = rewards[:, 2::3, :]
        # rewards = th.cat([rewards0, rewards1, rewards2], dim=0)
        # mask0 = mask[:, ::3, :]
        # mask1 = mask[:, 1::3, :]
        # mask2 = mask[:, 2::3, :]
        # mask = th.cat([mask0, mask1, mask2], dim=0)
        # terminated0 = terminated[:, ::3, :]
        # terminated1 = terminated[:, 1::3, :]
        # terminated2 = terminated[:, 2::3, :]
        # terminated = th.cat([terminated0, terminated1, terminated2], dim=0)

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        chosen_action_qvals = th.zeros([batch.batch_size*self.n_agents, (batch.max_seq_length)//self.n_agents , 1], device=self.device, dtype=th.float32)
        cur_max_actions = th.zeros([batch.batch_size, (batch.max_seq_length) , 1],  device=self.device, dtype=th.int)
        for t in range(max([len([x for x in ep if x]) for ep in batch["obs"]])-1):
            agent_outs = self.mac.forward(batch, t=t)
            avail_q_mask = th.tensor([
                0 if len(ep[t]) == 0 else 1 for ep in batch["avail_actions"]
            ], dtype=th.float32, device="cuda")
            mask_indices = (avail_q_mask == 1).nonzero(as_tuple=False).squeeze(1)
            for i, idx in enumerate(mask_indices):
                q_values = agent_outs[i]
                action_idx = actions[idx][t]
                assert action_idx < len(q_values)
                chosen_action_qvals[(t % self.n_agents) * batch.batch_size + idx][t//self.n_agents] = q_values[action_idx]
                cur_max_actions[idx][t] = q_values.max(dim=0, keepdim=True)[1]


        target_max_qvals = th.zeros([batch.batch_size*self.n_agents, (batch.max_seq_length)//self.n_agents , 1], device=self.device,
                                    dtype=th.float32)
        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(max([len([x for x in ep if x]) for ep in batch["obs"]])-1):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                avail_q_mask = th.tensor([
                    0 if len(ep[t]) == 0 else 1 for ep in batch["avail_actions"]
                ], dtype=th.float32, device="cuda")
                mask_indices = (avail_q_mask == 1).nonzero(as_tuple=False).squeeze(1)
                for i, idx in enumerate(mask_indices):
                    q_values = target_agent_outs[i]
                    action_idx = cur_max_actions[idx][t]
                    assert action_idx < len(q_values)
                    target_max_qvals[(t % self.n_agents) * batch.batch_size + idx][t // self.n_agents] = q_values[action_idx]
                target_mac_out.append(target_agent_outs)




            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_rlcard_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # # Mixer
        # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)


        mask = mask.expand_as(td_error2)
        # l = len(td_error2)//3
        # td_error2 = td_error2[l:,:,:]
        # mask = mask[l:,:,:]
        mask.unsqueeze(2)
        masked_td_error = td_error2

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets* mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env


        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
