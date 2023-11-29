import copy
from typing import List, Dict, Any
from enum import Enum
import numpy as np

from .bandit import NonStationaryMultiArmedBandit


class AgentType(Enum):
    VOTER_MODEL = 'VOTER_MODEL'


class Agent:
    def __init__(self, aid: int, atype: AgentType, initial_action: int = None):
        self.aid = aid
        self.atype = atype
        self.payoff_history = []
        self.action_history = []
        self.next_action = initial_action

    def prepare_step(self, step_num: int, nbrs: List['Agent'], variable_mab: NonStationaryMultiArmedBandit) -> None:
        pass

    def step(self, step_num: int, variable_mab: NonStationaryMultiArmedBandit) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {'aid': self.aid, 'atype': self.atype.value}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()


# Voter model agent
# -----------------

class VoterModelAgent(Agent):
    def __init__(self, aid: int, softmax_prob: float, memory_decay: float, initial_action: int = None):
        super().__init__(aid=aid, atype=AgentType.VOTER_MODEL, initial_action=initial_action)
        self.softmax_prob = softmax_prob
        self.memory_decay = memory_decay
        self.memory: Dict[int, float] = {}

    def prepare_step(self, step_num: int, nbrs: List[Agent], variable_mab: NonStationaryMultiArmedBandit) -> None:
        if step_num == 0:
            if self.next_action is None:
                self.next_action = np.random.choice(variable_mab.n_bandits(step_num=step_num))
            self.memory = {a: 1 for a in range(variable_mab.n_bandits(step_num=step_num))}
        else:
            if np.random.uniform() < self.softmax_prob:
                exp_payoff = np.exp(list(self.memory.values()))
                softmax_probs = exp_payoff / np.sum(exp_payoff)
                self.next_action = np.random.choice(list(self.memory.keys()), p=softmax_probs)
            else:
                random_nbr = np.random.randint(len(nbrs))
                self.next_action = nbrs[random_nbr].action_history[-1]

    def step(self, step_num: int, variable_mab: NonStationaryMultiArmedBandit) -> None:
        self.action_history.append(copy.deepcopy(self.next_action))
        payoff = variable_mab.pull(arm=self.next_action, step_num=step_num)
        self.payoff_history.append(payoff)
        self._update_memory(payoff=payoff)
        self.next_action = None

    def _update_memory(self, payoff: float) -> None:
        if self.next_action not in self.memory:
            self.memory[self.next_action] = payoff
        else:
            self.memory[self.next_action] = self.memory[self.next_action] * self.memory_decay + payoff * (1 - self.memory_decay)

    def to_dict(self) -> Dict[str, Any]:
        return {'aid': self.aid, 'atype': self.atype.value, 'softmax_prob': self.softmax_prob, 'memory_decay': self.memory_decay}
