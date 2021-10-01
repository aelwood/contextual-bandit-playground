from __future__ import annotations

import abc
import numpy as np
import typing

from abc import ABC

from sklearn.datasets import make_blobs
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
