from typing import List

import networkx as nx

from .agent import AgentType, Agent, VoterModelAgent
from .bandit import VariableMultiArmedBandit
from .experiment import Experiment


def run_experiment(eid: int, n_steps: int, agents: List[Agent], variable_mab: VariableMultiArmedBandit, network: nx.Graph) -> Experiment:
    experiment = Experiment(eid=eid, agents=agents, variable_multi_armed_bandit=variable_mab, network=network)
    experiment.run_experiment(n_steps=n_steps)
    return experiment


def get_agents(n_agents: int, agent_type: AgentType, **kwargs) -> List[Agent]:
    if agent_type == AgentType.VOTER_MODEL:
        return [VoterModelAgent(aid=aid, **kwargs) for aid in range(n_agents)]
    else:
        raise NotImplementedError(f'Agent type {agent_type} not implemented.')