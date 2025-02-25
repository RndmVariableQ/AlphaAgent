from abc import ABC, abstractmethod

from alphaagent.core.experiment import Task, Workspace
from alphaagent.core.scenario import Scenario


class Feedback:
    pass


class Evaluator(ABC):
    def __init__(
        self,
        scen: Scenario,
    ) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError
