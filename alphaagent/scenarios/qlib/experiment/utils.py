import io
import re
import shutil
from pathlib import Path

import pandas as pd

# render it with jinja
from jinja2 import Environment, StrictUndefined

from alphaagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from alphaagent.utils.env import QTDockerEnv
from alphaagent.log import logger


def generate_data_folder_from_qlib(use_local: bool = True):
    template_path = Path(__file__).parent / "factor_data_template"
    qtde = QTDockerEnv(is_local=use_local)
    qtde.prepare()
    
    # 运行数据生成脚本
    logger.info(f"在{'本地' if use_local else 'Docker容器'}中生成因子数据")
    execute_log = qtde.run(
        local_path=str(template_path),
        entry=f"python generate.py",
    )

    # 检查文件是否生成
    daily_pv_all = Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5"
    daily_pv_debug = Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5"
    
    assert daily_pv_all.exists(), "daily_pv_all.h5 is not generated."
    assert daily_pv_debug.exists(), "daily_pv_debug.h5 is not generated."

    # 创建数据目录并复制文件
    logger.info(f"复制生成的数据文件到工作目录")
    Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        daily_pv_all,
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md",
    )

    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        daily_pv_debug,
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md",
    )
    
    logger.info(f"数据准备完成")
    


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string(
        """
{{file_name}}
```{{type_desc}}
{{content}}
```
"""
    )

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        # get df.head() as string with full width
        pd.set_option("display.max_columns", None)  # or 1000
        pd.set_option("display.max_rows", None)  # or 1000
        pd.set_option("display.max_colwidth", None)  # or 199

        if isinstance(df.index, pd.MultiIndex):
            df_info = f"MultiIndex names:, {df.index.names})\n"
        else:
            df_info = f"Index name: {df.index.name}\n"
        columns = df.dtypes.to_dict()
        filtered_columns = [f"{i, j}" for i, j in columns.items() if i in variable_list]
        if filtered_columns:
            df_info += "Related Data columns: \n"
            df_info += ",".join(filtered_columns)
        else:
            df_info += "Data columns: \n"
            df_info += ",".join(columns)
        df_info += "\n"
        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += f"""
A snapshot of one instrument, from which you can tell the distribution of the data:
{df_on_one_instrument.head(5)}
"""
        return JJ_TPL.render(
            file_name=p.name,
            type_desc="h5 info",
            content=df_info,
        )
    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="markdown",
                content=content,
            )
    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(
    fname_reg: str = ".*",
    flags=0,
    variable_mapping=None,
    use_local: bool = True,
) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib(use_local=use_local)
    content_l = []
    
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)
