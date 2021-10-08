from __future__ import annotations

import abc
import typing
import scipy

from abc import ABC

from scipy.stats import norm

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type, Union


class PolicyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def notify_event(self, context, reward):
        pass

    @abc.abstractmethod
    def get_action(self, context):
        pass


class RandomPolicy(PolicyABC, ABC):
    def __init__(self,distribution):
        assert type(distribution) == scipy.stats._distn_infrastructure.rv_frozen
        self.distribution = distribution

    def train(self):
        pass

    def notify_event(self, context, reward):
        pass

    def get_action(self, context):
        return self.distribution.rvs()