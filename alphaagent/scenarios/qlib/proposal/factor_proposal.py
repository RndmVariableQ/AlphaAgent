import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from alphaagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from alphaagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from alphaagent.core.prompts import Prompts
from alphaagent.core.proposal import Hypothesis, Scenario, Trace
from alphaagent.core.experiment import Experiment
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from alphaagent.oai.llm_utils import APIBackend
import os
import pandas as pd
from alphaagent.log import logger
from alphaagent.scenarios.qlib.regulator.factor_regulator import FactorRegulator

QlibFactorHypothesis = Hypothesis
alphaagent_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts_alphaagent.yaml")

class AlphaAgentHypothesis(Hypothesis):
    """
    AlphaAgentHypothesis extends the Hypothesis class to include a potential_direction,
    which represents the initial idea or starting point for the hypothesis.
    """

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str
    ) -> None:
        super().__init__(
            hypothesis,
            "",
            "",
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        
    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

class QlibFactorHypothesisGen(FactorHypothesisGen):
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
            "RAG": None,
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["factor_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            # expression = response_dict[factor_name]["expression"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    # factor_expression=expression,
                    variables=variables,
                )
            )

        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp



alphaagent_prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts_alphaagent.yaml")

# prompt_dict不能作为属性，因为后续整个类的实例要被转为pickle，而prompt_dict不能转
class AlphaAgentHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str=None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.potential_direction = potential_direction

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        
        if len(trace.hist) > 0:
            hypothesis_and_feedback = (
                    Environment(undefined=StrictUndefined)
                    .from_string(alphaagent_prompt_dict["hypothesis_and_feedback"])
                    .render(trace=trace)
                )
            
        elif self.potential_direction is not None: 
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(alphaagent_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            ) # 
        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": alphaagent_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": alphaagent_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        response_dict = json.loads(response)
        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict["hypothesis"],
            concise_observation=response_dict["concise_observation"],
            concise_knowledge=response_dict["concise_knowledge"],
            concise_justification=response_dict["concise_justification"],
            concise_specification=response_dict["concise_specification"],
        )
        return hypothesis
    
    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
                round=len(trace.hist)
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        hypothesis = self.convert_response(resp)

        return hypothesis
    
    

class EmptyHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)
        
    def convert_response(self, *args, **kwargs) -> AlphaAgentHypothesis: 
        return super().convert_response(*args, **kwargs)  
    
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:

        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )

        return hypothesis




class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_regulator = FactorRegulator()
        
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = alphaagent_prompt_dict["factor_experiment_output_format"]
        function_lib_description = alphaagent_prompt_dict['function_lib_description']
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(alphaagent_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.background, # get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                function_lib_description=context["function_lib_description"],
                target_list=context["target_list"],
                RAG=context["RAG"], 
                expression_duplication=None
            )
        )
        
        # Detect duplicated sub-expressions
        flag = False
        expression_duplication_prompt = None
        while True:
            if flag:
                break
                
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
            response_dict = json.loads(resp)
            proposed_names = []
            proposed_exprs = []
            
            for i, factor_name in enumerate(response_dict):
                expr = response_dict[factor_name]["expression"]
                
                # Check if expression is parsable
                if not self.factor_regulator.is_parsable(expr):
                    logger.info(f"Failed to parse expr: {expr}, retrying...")
                    break
                
                success, eval_dict = self.factor_regulator.evaluate(expr)
                if not success:
                    break
                
                # If expression has problems, regenerate with feedback
                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = '\n\n'.join([expression_duplication_prompt, 
                        (Environment(undefined=StrictUndefined)
                            .from_string(alphaagent_prompt_dict["expression_duplication"])
                            .render(
                                prev_expression=expr,
                                duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                                duplicated_subtree=eval_dict['duplicated_subtree']
                            )
                        )])
                    else:
                        expression_duplication_prompt = (
                            Environment(undefined=StrictUndefined)
                            .from_string(alphaagent_prompt_dict["expression_duplication"])
                            .render(
                                prev_expression=expr,
                                duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                                duplicated_subtree=eval_dict['duplicated_subtree']
                            )
                        )
                    
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(alphaagent_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"], 
                            expression_duplication=expression_duplication_prompt
                        )
                    )
                    break
                else:
                    proposed_names.append(factor_name)
                    proposed_exprs.append(expr)
                    if i == len(response_dict) - 1:
                        flag = True
                    else:
                        continue
        

        # Add valid factors to the factor regulator
        self.factor_regulator.add_factor(proposed_names, proposed_exprs)
                
                
        return self.convert_response(resp, trace)
    

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            expression = response_dict[factor_name]["expression"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables,
                )
            )
            
        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp



class BacktestHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, factor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_path = factor_path
        
    def convert_response(self, *args, **kwargs) -> FactorExperiment:
        return super().convert_response(*args, **kwargs)
        
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        if os.path.exists(self.factor_path):
            tasks = []
            factor_df = pd.read_csv(self.factor_path, index_col=None)
            for index, (name, expr) in factor_df.iterrows():
                tasks.append(
                    FactorTask(
                        factor_name=name,
                        factor_description="",
                        factor_formulation="",
                        factor_expression=expr,
                        variables="",
                    )
                )
            
            exp = QlibFactorExperiment(tasks)
            exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

            unique_tasks = []

            for task in tasks:
                duplicate = False
                for based_exp in exp.based_experiments:
                    for sub_task in based_exp.sub_tasks:
                        if task.factor_name == sub_task.factor_name:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if not duplicate:
                    unique_tasks.append(task)

            exp.tasks = unique_tasks
            return exp
            
        else:
            raise ValueError(f"File {self.factor_csv_path} does not exist. ")
        
    