import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from alphaagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from alphaagent.components.proposal import ModelHypothesis2Experiment, ModelHypothesisGen
from alphaagent.core.prompts import Prompts
from alphaagent.core.proposal import Hypothesis, Scenario, Trace
from alphaagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

QlibModelHypothesis = Hypothesis


class QlibModelHypothesisGen(ModelHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": "In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.",
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": prompt_dict["model_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibModelHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
        )
        return hypothesis


class QlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["model_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[ModelExperiment] = [t[1] for t in trace.hist]

        model_list = []
        for experiment in experiment_list:
            model_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": model_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    model_type=model_type,
                )
            )
        exp = QlibModelExperiment(tasks)
        exp.based_experiments = [t[1] for t in trace.hist if t[2]]
        return exp
