<h4 align="center">
  <img src="docs/_static/logo.png" alt="RA-Agent logo" style="width:70%; ">
  
  <!-- <a href="https://arxiv.org/abs/2502.16789"><b>📃Paper Link</b>👁️</a> -->
</h3>




# 📖Introduction
<div align="center">
      <img src="docs/_static/workflow.png" alt="Our focused scenario" style="width:60%; ">
</div>



<!-- Tag Cloud -->
**AlphaAgent** is an autonomous framework that effectively integrates LLM agents for mining interpretable and decay-resistant alpha factors through three specialized agents.  

- **Idea Agent**: Proposes market hypotheses to guide factor creation based on financial theories or emerging trends.  
- **Factor Agent**: Constructs factors based on hypotheses while incorporating regularization mechanisms to avoid duplication and overfitting. 
- **Eval Agent**: Validates practicality, performs backtesting, and iteratively refines factors via feedback loops.

This repository follows the implementation of [RD-Agent](https://github.com/microsoft/RD-Agent). You can find its repository at: [https://github.com/microsoft/RD-Agent](https://github.com/microsoft/RD-Agent). We would like to extend our sincere gratitude to the RD-Agent team for their pioneering work and contributions to the community.


# ⚡ Quick start

### 🐍 Create a Conda Environment
- Create a new conda environment with Python (3.10 and 3.11 are well-tested in our CI):
  ```sh
  conda create -n alphaagent python=3.10
  ```
- Activate the environment:
  ```sh
  conda activate alphaagent
  ```

### 🛠️ Install locally
- 
  ```sh
  # Install AlphaAgent
  pip install -e .
  ```

### 📈 Data Preparation
- First, clone Qlib source code for runing backtest locally.
  ```
  # Clone Qlib source code
  git clone https://github.com/microsoft/qlib.git
  cd qlib
  pip install .
  cd ..
  ```

- Then, mannully download Chinese stock data via baostock and dump into the Qlib format.
  ```sh
  # Download or update stock data from 2015-01-01 until NOW from baostock
  python prepare_cn_data.py

  cd qlib

  # Convert csv to Qlib format. Check correct paths before runing. 
  python scripts/dump_bin.py dump_all ... \
  --include_fields open,high,low,close,preclose,volume,amount,turn,factor \
  --csv_path  ~/.qlib/qlib_data/cn_data/raw_data_now \
  --qlib_dir ~/.qlib/qlib_data/cn_data \
  --date_field_name date \
  --symbol_field_name code

  # Collect calendar data
  python scripts/data_collector/future_calendar_collector.py --qlib_dir ~/.qlib/qlib_data/cn_data/ --region cn


  # Download the CSI500/CSI300/CSI100 stock universe
  python scripts/data_collector/cn_index/collector.py --index_name CSI500 --qlib_dir ~/.qlib/qlib_data/cn_data/ --method parse_instruments
  ```


- Alternatively, stock data (out-dated) will be automatically downloaded to `~/.qlib/qlib_data/cn_data`.


- You can modify backtest configuration files which are located at:
  - Baseline: `alphaagent/scenarios/qlib/experiment/factor_template/conf.yaml`
  - For Newly proposed factors: `alphaagent/scenarios/qlib/experiment/factor_template/conf_cn_combined.yaml`
  - For changing train/val/test periods, first remove all cache files in `./git_ignore_folder` and `./pickle_cache`. 
  - For changing the market, remove cache files in `./git_ignore_folder`, `./pickle_cache`. Then, delete `daily_pv_all.h5` and `daily_pv_debug.h5` in directory `alphaagent/scenarios/qlib/experiment/factor_data_template/`. 


### 💊 Health check
- We provide a health check that currently checks two things.
  - whether the docker installation was successful.
  - whether the default port is occupied.
  ```sh
  alphaagent health_check
  ```


### ⚙️ Configuration
- For OpenAI compatible API, ensure both `OPENAI_BASE_URL` and `OPENAI_API_KEY` are configured in the `.env` file.
- add `USE_LOCAL=True` in `.env` file if you want to run this project using local environment instead of docker


### 🚀 Run AlphaAgent
- Run **AlphaAgent** based on [Qlib Backtesting Framework](http://github.com/microsoft/qlib).
  ```sh
  alphaagent mine --potential_direction "<YOUR_MARKET_HYPOTHESIS>"
  ```

- Alternatively, run the following command
  ```sh
  dotenv run -- python alphaagent/app/qlib_rd_loop/factor_alphaagent.py --direction "<YOUR_MARKET_HYPOTHESIS>"
  ```
  After running the command, log out and log back in for the changes to take effect. 

- Multi-factor backtesting
  ```sh
  alphaagent backtest --factor_path "<PATH_TO_YOUR_CSV_FILE>"
  ```

  Your factors need to be stored in a `.csv` file. Here is an example:
  ```csv
  factor_name,factor_expression
  MACD_Factor,"MACD($close)"
  RSI_Factor,"RSI($close)"
  ```


- If you need to rerun the baseline results or update backtest configs, remove the cache folders:
  ```sh
  rm -r ./pickle_cache/*
  rm -r ./git_ignore_folder/*
  ```

### 🖥️ Monitor the Application Results
- You can run the following command for our demo program to see the run logs.

  ```sh
  alphaagent ui --port 19899 --log_dir log/
  ```

### 📊 Performance
<div align="center">
      <img src="docs/_static/excess_return.png" alt="Excess Return Comparison" style="width:90%; ">
</div>

The above figure illustrates our method's excess return performance against the CSI 500 benchmark over the past four years. The results demonstrate the remarkable effectiveness of our approach in generating stable alpha factors, consistently outperforming not only traditional methods but also cutting-edge reasoning models like OpenAI-o1 and DeepSeek-R1 that are currently in the spotlight. 

### 🤖 Prompt Files

#### 💡 IdeaAgent's Prompts: 
- `alphaagent/scenarios/qlib/prompts_alphaagent.yaml`

#### ⚙️ FactorAgent's Prompts: 
- `alphaagent/components/coder/factor_coder/prompts_alphaagent.yaml`



