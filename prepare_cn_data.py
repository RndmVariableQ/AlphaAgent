import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np

def get_all_stocks_in_period(start_date, end_date):
    """获取指定时间段内所有出现过的股票代码"""
    all_stocks = set()
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    current = start
    while current <= end:
        query_date = current.strftime('%Y-%m-%d')
        stock_rs = bs.query_all_stock(query_date)
        stock_df = stock_rs.get_data()
        if not stock_df.empty:
            all_stocks.update(stock_df['code'].tolist())
        current += relativedelta(years=1)
        if current > end:
            break
    print(f"共获取到 {len(all_stocks)} 只股票")
    return all_stocks

def download_stock_data(start_date, end_date, output_dir):
    """下载或更新股票数据到最新日期"""
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return
    
    try:
        all_stocks = get_all_stocks_in_period(start_date, end_date)
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def download_single_stock(code):
            code_clean = code.replace('.', '')
            output_file = output_path / f"{code_clean}.csv"
            
            # 确定该股票的下载起始日期
            if output_file.exists():
                existing_df = pd.read_csv(output_file)
                if not existing_df.empty:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    last_date = existing_df['date'].max()
                    code_download_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    # 如果无需更新则跳过
                    # print(f"股票 {code} 已下载开始日期：{code_download_start_date}，结束日期：{last_date.strftime('%Y-%m-%d')}")
                    if code_download_start_date == end_date:
                        print(f"股票 {code} 无需更新")
                        return
                else:
                    code_download_start_date = start_date
            else:
                code_download_start_date = start_date
            
            # 下载增量数据
            print(f"下载 {code} 数据...日期范围：{code_download_start_date} 至 {end_date}")
            rs = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=code_download_start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="1" # 后复权
            )
            
            if rs.error_code != '0':
                print(f"获取 {code} 数据失败: {rs.error_msg}")
                return
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())



            # 获取复权因子
            rs_list = []
            rs_adj = bs.query_adjust_factor(
                code,
                start_date=code_download_start_date,
                end_date=end_date,
            )

            while (rs_adj.error_code == '0') & rs_adj.next():
                rs_list.append(rs_adj.get_row_data())

            adj_df = pd.DataFrame(rs_list, columns=rs_adj.fields).set_index('dividOperateDate')['adjustFactor']
            adj_df = adj_df.rename('factor')

            
            if data_list:
                new_df = pd.DataFrame(data_list, columns=rs.fields).set_index('date')
                new_df = pd.concat([new_df, adj_df], axis=1).ffill().fillna(1)
                
                new_df['code'] = new_df['code'].str.replace('.', '', regex=False)
                # new_df['factor'] = np.ones(len(new_df))
                numeric_cols = new_df.columns[2:]
                new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                new_df = new_df.reset_index()
                new_df = new_df.rename(columns={'index': 'date'})
                
                # 合并并保存数据
                if output_file.exists():
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date', 'code'])
                    combined_df['date'] = pd.to_datetime(combined_df['date'])
                    combined_df = combined_df.sort_values('date')
                else:
                    combined_df = new_df
                
                
                combined_df.to_csv(output_file, index=False, encoding='utf-8')
            
            # time.sleep(0.5)
        
        # # 使用线程池进行并发下载
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(download_single_stock, code) for code in all_stocks]
            
            # 使用tqdm显示进度
            for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
                pass

        # for code in all_stocks:
        #     download_single_stock(code)
                
    finally:
        bs.logout()



def download_oneday_stock_data_(date):

    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取某日所有证券信息 ####
    rs = bs.query_all_stock(day=date)
    print('query_all_stock respond error_code:'+rs.error_code)
    print('query_all_stock respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    # result.to_csv("D:\\all_stock.csv", encoding="gbk", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()

if __name__ == '__main__':
    # 动态设置结束日期为当前日期
    START_DATE = '2014-12-31'
    END_DATE = (datetime.now()).strftime('%Y-%m-%d') # '2025-01-01'  - timedelta(days=7)
    DATA_DIR = '~/.qlib/qlib_data/cn_data/raw_data_back_adjust'
    
    print("开始下载股票数据...日期范围：", START_DATE, "至", END_DATE)
    download_stock_data(START_DATE, END_DATE, DATA_DIR)
    # download_oneday_stock_data_((datetime.now()-timedelta(days=1)).strftime('%Y-%m-%d'))
    print("下载完成!")