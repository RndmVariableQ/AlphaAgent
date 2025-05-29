from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.core.experiment import FBWorkspace
from alphaagent.log import logger
from alphaagent.utils.env import QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(
        self, 
        qlib_config_name: str = "conf.yaml", 
        run_env: dict = {}, 
        use_local: bool = True, 
        *args, 
        **kwargs
    ) -> str:
        # 使用本地环境或Docker环境
        qtde = QTDockerEnv(is_local=use_local)
        qtde.prepare()
        
        # 运行Qlib回测
        logger.info(f"Execute {'Local' if use_local else 'Docker container'} Backtest: qrun {qlib_config_name}")
        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )

        # 处理结果
        logger.info(f"Read {'Local' if use_local else 'Docker container'} Backtest Result")
        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        # 加载结果
        ret_df = pd.read_pickle(self.workspace_path / "ret.pkl")
        logger.log_object(ret_df, tag="Quantitative Backtesting Chart")

        csv_path = self.workspace_path / "qlib_res.csv"
        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None

        return pd.read_csv(csv_path, index_col=0).iloc[:, 0]
