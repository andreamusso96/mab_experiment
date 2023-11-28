import copy
from typing import List, Dict, Any
from enum import Enum
import numpy as np

from .bandit import VariableMultiArmedBandit


class AgentType(Enum):
    VOTER_MODEL = 'VOTER_MODEL'


class Agent:
    def __init__(self, aid: int, atype: AgentType):
        self.aid = aid
        self.atype = atype
        self.payoff_history = []
        self.action_history = []
        self.next_action = None

    def prepare_step(self, step_num: int, nbrs: List['Agent'], variable_mab: VariableMultiArmedBandit) -> None:
        pass

    def step(self, step_num: int, variable_mab: VariableMultiArmedBandit) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {'aid': self.aid, 'atype': self.atype.value}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()


class VoterModelAgent(Agent):
    def __init__(self, aid: int, softmax_prob: float, memory_window: int):
        super().__init__(aid=aid, atype=AgentType.VOTER_MODEL)
        self.softmax_prob = softmax_prob
        self.memory_window = memory_window

    def prepare_step(self, step_num: int, nbrs: List[Agent], variable_mab: VariableMultiArmedBandit) -> None:
        if step_num == 0:
            self.next_action = np.random.choice(variable_mab.n_bandits(step_num=step_num))
        else:
            if np.random.uniform() < self.softmax_prob:
                exp_payoff = np.exp(self.payoff_history[-self.memory_window:])
                softmax_probs = exp_payoff / np.sum(exp_payoff)
                self.next_action = np.random.choice(self.action_history[-self.memory_window:], p=softmax_probs)
            else:
                random_nbr = np.random.randint(len(nbrs))
                self.next_action = nbrs[random_nbr].action_history[-1]

    def step(self, step_num: int, variable_mab: VariableMultiArmedBandit) -> None:
        self.action_history.append(copy.deepcopy(self.next_action))
        payoff = variable_mab.pull(arm=self.next_action, step_num=step_num)
        self.payoff_history.append(payoff)
        self.next_action = None

    def to_dict(self) -> Dict[str, Any]:
        return {'aid': self.aid, 'atype': self.atype.value, 'softmax_prob': self.softmax_prob, 'memory_window': self.memory_window}

