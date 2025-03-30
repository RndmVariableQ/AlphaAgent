"""
Factor workflow with session control
"""

from typing import Any
import fire
import signal
import sys
import threading
from functools import wraps
import time
import ctypes
import os
from alphaagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
from alphaagent.components.workflow.alphaagent_loop import AlphaAgentLoop
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.log.time import measure_time
from alphaagent.oai.llm_conf import LLM_SETTINGS




def force_timeout():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 优先选择timeout参数
            seconds = LLM_SETTINGS.factor_mining_timeout
            def handle_timeout(signum, frame):
                logger.error(f"强制终止程序执行，已超过{seconds}秒")
                sys.exit(1)

            # 设置信号处理器
            signal.signal(signal.SIGALRM, handle_timeout)
            # 设置闹钟
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # 取消闹钟
                signal.alarm(0)
            return result
        return wrapper
    return decorator


@force_timeout()
def main(path=None, step_n=None, direction=None, stop_event=None):
    """
    Autonomous alpha factor mining. 

    Args:
        path: 会话路径
        step_n: 步骤数
        direction: 初始方向
        stop_event: 停止事件
        timeout: 超时时间（秒），默认12小时

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_alphaagent.py $LOG_PATH/__session__/1/0_propose  --step_n 1  --potential_direction "[Initial Direction (Optional)]"  # `step_n` is a optional paramter

    """
    try:
        if path is None:
            model_loop = AlphaAgentLoop(ALPHA_AGENT_FACTOR_PROP_SETTING, potential_direction=direction, stop_event=stop_event)
        else:
            model_loop = AlphaAgentLoop.load(path)
        model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise
    finally:
        logger.info("程序执行完成或被终止")

if __name__ == "__main__":
    fire.Fire(main)
