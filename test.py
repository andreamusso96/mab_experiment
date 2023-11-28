from typing import List

import networkx as nx

import mab_experiment as exp


def get_variable_multi_armed_bandit(mab_start_rounds: List[int]) -> exp.VariableMultiArmedBandit:
    mean_std_1 = [(1, 0.5), (1.2, 0.5), (1.4 , 0.5), (1.6, 0.5)]
    bandits_1 = [exp.NormalDistributionBandit(bid=i, mean=mean, std=std) for i, (mean, std) in enumerate(mean_std_1)]
    multi_armed_bandit_1 = exp.MultiArmedBandit(bandits=bandits_1)

    mean_std_2 = [(1.6, 0.5), (1.5, 0.5), (1.2 , 0.5), (1, 0.5)]
    bandits_2 = [exp.NormalDistributionBandit(bid=i, mean=mean, std=std) for i, (mean, std) in enumerate(mean_std_2)]
    multi_armed_bandit_2 = exp.MultiArmedBandit(bandits=bandits_2)

    variable_mab = exp.VariableMultiArmedBandit(multi_armed_bandits=[multi_armed_bandit_1, multi_armed_bandit_2], mab_start_rounds=mab_start_rounds)
    return variable_mab


def get_agents(n_agents: int) -> List[exp.Agent]:
    return exp.get_agents(n_agents=n_agents, agent_type=exp.AgentType.VOTER_MODEL, softmax_prob=0.1)


def get_network(n_nodes: int, avg_degree: int, rewiring_prob: float):
    network = nx.watts_strogatz_graph(n=n_nodes, k=avg_degree, p=rewiring_prob)
    return network


def run_test():
    n_ag = 50
    n_steps = 40
    agents = get_agents(n_agents=n_ag)
    variable_mab = get_variable_multi_armed_bandit(mab_start_rounds=[0, 20])
    networkx = get_network(n_nodes=n_ag, avg_degree=4, rewiring_prob=0.1)
    experiment = exp.run_experiment(eid=0, n_steps=n_steps, agents=agents, variable_mab=variable_mab, network=networkx)
    experiment.save(folder_path='/Users/andrea/Desktop/PhD/Packages/Utils/mab_experiment/temp')


if __name__ == '__main__':
    run_test()
