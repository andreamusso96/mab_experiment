from typing import List

import networkx as nx
import pandas as pd

from .agent import Agent
from .bandit import VariableMultiArmedBandit


class Experiment:
    def __init__(self, eid: int, agents: List[Agent], variable_multi_armed_bandit: VariableMultiArmedBandit, network: nx.Graph):
        self.eid = eid
        self.agents = {agent.aid: agent for agent in agents}
        self.variable_multi_armed_bandit = variable_multi_armed_bandit
        self.network = network
        self.step_num = 0

    def run_experiment(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def step(self) -> None:
        for agent_id, agent in self.agents.items():
            nbrs = [self.agents[nbr] for nbr in self.network.neighbors(agent_id)]
            agent.prepare_step(step_num=self.step_num, nbrs=nbrs, variable_mab=self.variable_multi_armed_bandit)

        for agent_id, agent in self.agents.items():
            agent.step(step_num=self.step_num, variable_mab=self.variable_multi_armed_bandit)

        self.step_num += 1

    def save(self, folder_path: str) -> None:
        self._get_action_history().to_csv(f'{folder_path}/experiment_{self.eid}_action_history.csv')
        self._get_payoff_history().to_csv(f'{folder_path}/experiment_{self.eid}_payoff_history.csv')
        self.variable_multi_armed_bandit.to_pandas().to_csv(f'{folder_path}/experiment_{self.eid}_variable_multi_armed_bandit.csv')
        nx.to_pandas_adjacency(self.network).to_csv(f'{folder_path}/experiment_{self.eid}_network_adj.csv')
        self._get_agent_metadata().to_csv(f'{folder_path}/experiment_{self.eid}_agent_metadata.csv')

    def _get_action_history(self) -> pd.DataFrame:
        action_histories = pd.DataFrame([agent.action_history for agent in self.agents.values()], index=list(self.agents.keys()), columns=list(range(self.step_num))).T
        action_histories.index.name = 'round_num'
        return action_histories

    def _get_payoff_history(self) -> pd.DataFrame:
        payoff_histories = pd.DataFrame([agent.payoff_history for agent in self.agents.values()], index=list(self.agents.keys()), columns=list(range(self.step_num))).T
        payoff_histories.index.name = 'round_num'
        return payoff_histories

    def _get_agent_metadata(self) -> pd.DataFrame:
        agent_metadata = pd.DataFrame([agent.to_dict() for agent in self.agents.values()])
        return agent_metadata
