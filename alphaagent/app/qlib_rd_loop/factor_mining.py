"""
Factor workflow with session control
"""

from typing import Any

import fire

from alphaagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
from alphaagent.components.workflow.alphaagent_loop import AlphaAgentLoop
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.log.time import measure_time


def main(path=None, step_n=None, direction=None, stop_event=None):
    """
    Autonomous alpha factor mining. 

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_alphaagent.py $LOG_PATH/__session__/1/0_propose  --step_n 1  --potential_direction "[Initial Direction (Optional)]"  # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = AlphaAgentLoop(ALPHA_AGENT_FACTOR_PROP_SETTING, potential_direction=direction, stop_event=stop_event)
    else:
        model_loop = AlphaAgentLoop.load(path)
    model_loop.run(step_n=step_n, stop_event=stop_event)


if __name__ == "__main__":
    fire.Fire(main)
