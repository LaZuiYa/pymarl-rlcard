import random

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class RlCardRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "player_id": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["player_id"].append(0)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, model_palyer_id=None):
        if model_palyer_id is None:
            model_palyer_id = [0]
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)

        if test_mode:
            self.mac.model_player_id = model_palyer_id
        else:
            # self.mac.model_player_id = [1,2]
            if self.t_env <= 3000000:
                self.mac.model_player_id = [random.choice([0, 1, 2])] #np.random.choice([0,1,2],  p=[0.2, 0.4, 0.4])
            elif 3000000<self.t_env <= 5000000:
                self.mac.model_player_id = random.sample([0, 1, 2], 2)
            else:
                self.mac.model_player_id = [0,1,2]
        while True:

            # Pass the entire batch of experiences up till now to the agents.

            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         bs=envs_not_terminated, test_mode=test_mode)
            else:
                if any(terminated):
                    envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                                  test_mode=test_mode)

            cpu_actions = np.array(actions)

            action_indices = []
            for i, j in enumerate(envs_not_terminated):
                avail_actions_t = self.batch["avail_actions"][j]
                chosen_action = actions[i]
                idx = list(avail_actions_t[self.t].keys()).index(chosen_action)
                action_indices.append(idx)

            # Update the actions taken
            actions_chosen = {
                "actions": np.array(action_indices),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

            if envs_not_terminated == []:
                break


            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "player_id": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"] if data["player_id"] == 1 else 0   # reward for lardland
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    # if data["terminated"]:
                    #     final_env_infos.append(data["info"])
                    if data["terminated"] : # and not data["info"].get("episode_limit", False)
                        trace = data["state"]["trace"]

                        for tt in range(len(trace) - 3, -1, -1):
                            if tt % 3 == 0 and trace[tt+1][1]=='pass'and trace[tt+2][1] == 'pass':
                                self.batch["reward"][idx][tt] = 0.1
                            else:
                                for next_t in [tt + 1, tt + 2]:
                                    if next_t < len(trace) and next_t % 3 == 0:
                                        if trace[next_t][1] == 'pass':
                                            self.batch["reward"][idx][tt] = 0.1


                        env_terminated = True
                        if data["player_id"] == 1: # lardland win
                            self.batch["reward"][idx][self.t + 1] = -1
                            self.batch["terminated"][idx][self.t + 1] = 1
                            self.batch["filled"][idx][self.t + 1] = 1
                            self.batch["reward"][idx][self.t+2] = -1
                            self.batch["terminated"][idx][self.t+2] = 1
                            self.batch["filled"][idx][self.t+2] = 1
                            self.batch["filled"][idx][self.t+3] = 1
                        if data["player_id"] == 2:  # lardland win
                            self.batch["reward"][idx][self.t + 1] = 1
                            self.batch["terminated"][idx][self.t + 1] = 1
                            self.batch["filled"][idx][self.t + 1] = 1

                            self.batch["reward"][idx][self.t -1] = -1
                            self.batch["terminated"][idx][self.t - 1] = 1
                            self.batch["filled"][idx][self.t+2] = 1
                        if data["player_id"] == 0:  # lardland win
                            self.batch["reward"][idx][self.t - 1] = 1
                            self.batch["terminated"][idx][self.t - 1] = 1

                            self.batch["reward"][idx][self.t - 2] = -1
                            self.batch["terminated"][idx][self.t - 2] = 1
                            # self.batch["filled"][idx][self.t+2] = 1
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["player_id"].append(data["player_id"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run



        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_"+ str(model_palyer_id[0])  + "_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        if self.mac.model_player_id != [0]:
            episode_returns = [1 - x for x in episode_returns]

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            obs, terminated, player_id = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_perfect_information()
            avail_actions = env.get_avail_actions()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": env.get_payoffs()[(player_id+3-1)%3] if terminated else 0,
                "terminated": terminated,
                "player_id": player_id
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_perfect_information(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_state(0)
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

