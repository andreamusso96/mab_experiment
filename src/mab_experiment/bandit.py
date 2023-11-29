from typing import Dict, List, Any
from enum import Enum
import numpy as np
import pandas as pd


class BanditType(Enum):
    NORMAL_DISTRIBUTION = 'NORMAL_DISTRIBUTION'


class Bandit:
    def __init__(self, bid: int, btype: BanditType):
        self.bid = bid
        self.btype = btype

    def pull(self) -> float:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {'bid': self.bid, 'btype': self.btype.value}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()


class MultiArmedBandit:
    def __init__(self, bandits: List[Bandit]):
        self.bandits = {bandit.bid: bandit for bandit in bandits}

    def pull(self, arm: int) -> float:
        return self.bandits[arm].pull()

    def n_bandits(self) -> int:
        return len(self.bandits)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([bandit.to_dict() for bandit in self.bandits.values()])

    def __str__(self):
        return str({'n_bandits': self.n_bandits()})

    def __repr__(self):
        return self.__str__()


class NonStationaryMultiArmedBandit:
    def __init__(self, multi_armed_bandits: List[MultiArmedBandit], mab_start_steps: List[int]):
        self.multi_armed_bandits = multi_armed_bandits
        self.mab_start_steps = mab_start_steps

    def pull(self, arm: int, step_num: int) -> float:
        mab_id = np.searchsorted(self.mab_start_steps, step_num, side='right') - 1
        return self.multi_armed_bandits[mab_id].pull(arm=arm)

    def n_bandits(self, step_num: int) -> int:
        mab_id = np.searchsorted(self.mab_start_steps, step_num, side='right') - 1
        return self.multi_armed_bandits[mab_id].n_bandits()

    def to_pandas(self) -> pd.DataFrame:
        data = []
        for i, multi_armed_bandit in enumerate(self.multi_armed_bandits):
            mab_df = multi_armed_bandit.to_pandas()
            mab_df['mab_start_step'] = self.mab_start_steps[i]
            data.append(mab_df)
        return pd.concat(data)


class NormalDistributionBandit(Bandit):
    def __init__(self, bid: int, mean: float, std: float):
        super().__init__(bid=bid, btype=BanditType.NORMAL_DISTRIBUTION)
        self.mean = mean
        self.std = std

    def pull(self) -> float:
        return np.random.normal(loc=self.mean, scale=self.std)

    def to_dict(self) -> Dict[str, Any]:
        return {'bid': self.bid, 'btype': self.btype.value, 'mean': self.mean, 'std': self.std}
