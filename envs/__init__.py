from functools import partial
import sys
import os
from .multiagentenv import MultiAgentEnv


from rlcard.envs.doudizhu import DoudizhuEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}



REGISTRY["doudizhu"] = partial(env_fn, env=DoudizhuEnv)

