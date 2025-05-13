from copy import deepcopy
from pathlib import Path

from alphaagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from alphaagent.core.experiment import Task
from alphaagent.core.prompts import Prompts
from alphaagent.core.scenario import Scenario
from alphaagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from alphaagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

rdagent_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts_rdagent.yaml")


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")


class QlibFactorScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(rdagent_prompt_dict["qlib_factor_background"])
        self._source_data = deepcopy(get_data_folder_intro())
        self._output_format = deepcopy(rdagent_prompt_dict["qlib_factor_output_format"])
        self._interface = deepcopy(rdagent_prompt_dict["qlib_factor_interface"])
        self._strategy = deepcopy(rdagent_prompt_dict["qlib_factor_strategy"])
        self._simulator = deepcopy(rdagent_prompt_dict["qlib_factor_simulator"])
        self._rich_style_description = deepcopy(rdagent_prompt_dict["qlib_factor_rich_style_description"])
        self._experiment_setting = deepcopy(rdagent_prompt_dict["qlib_factor_experiment_setting"])

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        """A static scenario describer"""
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""



alphaagent_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts_alphaagent.yaml")
class QlibAlphaAgentScenario(Scenario):
    def __init__(self, use_local: bool = True) -> None:
        super().__init__()
        self._background = deepcopy(alphaagent_prompt_dict["qlib_factor_background"])
        self._source_data = deepcopy(get_data_folder_intro(use_local=use_local))
        self._output_format = deepcopy(alphaagent_prompt_dict["qlib_factor_output_format"])
        self._interface = deepcopy(alphaagent_prompt_dict["qlib_factor_interface"])
        self._strategy = deepcopy(alphaagent_prompt_dict["qlib_factor_strategy"])
        self._simulator = deepcopy(alphaagent_prompt_dict["qlib_factor_simulator"])
        self._rich_style_description = deepcopy(alphaagent_prompt_dict["qlib_factor_rich_style_description"])
        self._experiment_setting = deepcopy(alphaagent_prompt_dict["qlib_factor_experiment_setting"])

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        """A static scenario describer"""
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The simulator user can use to test your factor:
{self.simulator}
"""