"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import time
import pandas as pd
from typing import Any

from rdagent.components.workflow.conf import BaseFacSetting
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,  
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.time import measure_time
from rdagent.utils.workflow import LoopBase, LoopMeta
from rdagent.core.exception import FactorEmptyError

class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, potential_direction):
        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            ### 换成基于初始hypo的，生成完整的hypo
            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen, potential_direction)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            ### 换成一次生成10个因子
            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.factor_constructor, tag="experiment generation")

            ### 加入代码执行中的 Variables / Functions
            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    @measure_time
    def factor_propose(self, prev_out: dict[str, Any]):
        """
        提出作为构建因子的基础的假设
        """
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        return idea

    @measure_time
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        基于假设构造多个不同的因子
        """
        with logger.tag("r"):  
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    @measure_time
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        根据因子表达式计算过去的因子表（因子值）
        """
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor
    

    @measure_time
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        回测因子
        """
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["factor_calculate"])
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))

class BacktestLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, factor_csv_path=None):
        with logger.tag("init"):

            self.factor_csv_path = factor_csv_path

            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            ### 换成基于初始hypo的，生成完整的hypo
            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            ### 换成一次生成10个因子
            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.factor_constructor, tag="experiment generation")

            ### 加入代码执行中的 Variables / Functions
            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            # logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    @measure_time
    def factor_propose(self, prev_out: dict[str, Any]):
        """
        提出作为构建因子的基础的假设
        """
        with logger.tag("r"):  
            if self.factor_csv_path:
                # 从 CSV 文件中读取因子
                factors = pd.read_csv(self.factor_csv_path)
                factor_name = factors['factor_name'].iloc[0]  # 假设选取第一个因子
                factor_expression = factors['factor_expression'].iloc[0]  # 获取因子公式
                idea = {"factor_name": factor_name, "factor_expression": factor_expression}
                logger.log_object(idea, tag="factor from csv")
            else:
                # 如果没有 CSV 文件路径，则按原来的流程生成因子
                logger.warning("No CSV file path provided. Proceeding with original factor generation.")
        return idea

    @measure_time
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        跳过因子生成，直接使用 CSV 文件中的因子
        返回空值或一个默认的因子对象
        """
        with logger.tag("r"):  
            if self.factor_csv_path:
                # 返回一个默认的空因子对象或空值
                empty_factor = {
                    "sub_tasks": [],
                    "sub_workspace_list": [],
                }
                logger.log_object(empty_factor, tag="empty factor")
                return empty_factor
            else:
                factor = super().factor_construct(prev_out)
        return factor

    @measure_time
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        跳过因子计算，直接返回空值
        """
        with logger.tag("d"):  # develop
            if self.factor_csv_path:
                # 返回一个空的因子计算结果
                empty_factor = {
                    "sub_workspace_list": []
                }
                logger.log_object(empty_factor, tag="empty factor calculation")
                return empty_factor
            else:
                factor = super().factor_calculate(prev_out)
        return factor
    

    @measure_time
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        直接进行因子回测，跳过前面的因子计算步骤
        """
        with logger.tag("ef"):  # evaluate and feedback
            if self.factor_csv_path:
                # 如果是从 CSV 文件读取因子，直接返回一个默认的回测结果
                empty_backtest_result = {
                    "result": "no_backtest_data"
                }
                logger.log_object(empty_backtest_result, tag="empty backtest result")
                return empty_backtest_result
            else:
                exp = self.runner.develop(prev_out["factor_construct"])
                if exp is None:
                    logger.error(f"Factor extraction failed.")
                    raise FactorEmptyError("Factor extraction failed.")
                logger.log_object(exp, tag="runner result")
                return exp

    @measure_time
    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)
        """
        这里可以添加跳过反馈的条件
        """
        with logger.tag("ef"):  # evaluate and feedback
            # 判断是否需要跳过 feedback 步骤
            if self.skip_feedback:
                # 如果跳过，则返回一个默认的反馈值（可以是空字典或空列表等）
                feedback = {}
            else:
                feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
