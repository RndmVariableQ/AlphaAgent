"""
Factor workflow with session control
"""

from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
from rdagent.components.workflow.alphaagent_loop import AlphaAgentLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.log.time import measure_time


def main(path=None, step_n=None, potential_direction=None):
    """
    Auto R&D Evolving loop for fintech factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_alphaagent.py $LOG_PATH/__session__/1/0_propose  --step_n 1  --potential_direction "[Initial Direction (Optional)]"  # `step_n` is a optional paramter

    """
    if path is None:
        model_loop = AlphaAgentLoop(ALPHA_AGENT_FACTOR_PROP_SETTING, potential_direction)
    else:
        model_loop = AlphaAgentLoop.load(path)
    model_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
