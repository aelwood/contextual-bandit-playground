from __future__ import annotations

import abc
import typing

if typing.TYPE_CHECKING:
    pass


class EnvironmentABC(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_reward(self, action, context):
        pass

    @abc.abstractmethod
    def generate_contexts(self):
        pass

class SyntheticEnvironment(EnvironmentABC):
    pass
