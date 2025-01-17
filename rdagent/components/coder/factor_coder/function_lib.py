import numpy as np
import pandas as pd

def support_numpy(func):
    def wrapper(*args):
        # 对于典型输入，func(df, p) or func(df)
        if (len(args) == 2 and isinstance(args[0], np.ndarray) and not isinstance(args[1], np.ndarray)):
            # 转换NumPy数组到DataFrame
            new_args = (pd.DataFrame(args[0]), args[1])
            # 执行函数并转回NumPy数组
            result = func(*new_args)
            result = result.values
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            new_args = (pd.DataFrame(args[0]),)
            # 执行函数并转回NumPy数组
            result = func(*new_args)
            result = result.values
        elif len(args) == 1 and isinstance(args[0], int):
            new_args = (pd.DataFrame([args[0]]),)
            # 执行函数并转回NumPy数组
            result = func(*new_args)
            result = result.iloc[0, 0]
        else:
            result = func(*args)
        return result

    return wrapper

@support_numpy
def DELTA(df:pd.DataFrame, p:int=1):
    return df.groupby('instrument').transform(lambda x: x.diff(periods=p))

@support_numpy
def RANK(df:pd.DataFrame):
    return df.groupby('datetime').rank(pct=True)

@support_numpy
def TS_RANK(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).rank(pct=True))

@support_numpy
def TS_MAX(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).max())

@support_numpy
def TS_MIN(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).min())

@support_numpy
def MEAN(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())

@support_numpy
def MEDIAN(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).median())

@support_numpy
def SUM(df:pd.DataFrame, p:int=5):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

def MAX(x:pd.DataFrame, y:pd.DataFrame):
    return np.maximum(x, y)

def MIN(x:pd.DataFrame, y:pd.DataFrame):
    return np.minimum(x, y)

@support_numpy
def ABS(df:pd.DataFrame):
    return df.groupby('instrument').transform(lambda x: x.abs())

@support_numpy
def DELAY(df:pd.DataFrame, p:int=1):
    assert p >= 0, ValueError("DELAY的时长不能小于0，否则将会造成数据窥测")
    return df.groupby('instrument').transform(lambda x: x.shift(p))

def CORR(df1:pd.Series, df2: np.ndarray | pd.Series, p:int=5):
    if isinstance(df2, np.ndarray) and p != len(df2):
        p = len(df2)
        def corr(window):
            return np.correlate(window, df2[:len(window)])
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(corr, raw=True))
    else:
        def rolling_corr(group, df2, p):
            # 获取当前分组的 instrument
            instrument = group.name
            # 从 df2 中提取对应的 instrument 数据
            df2_group = df2.xs(instrument, level='instrument')
            # 计算滚动相关性
            return group.rolling(p, min_periods=2).corr(df2_group)

        # 使用 groupby 和 apply 来计算每个 instrument 的滚动相关性
        result = df1.groupby('instrument').apply(lambda x: rolling_corr(x, df2, p))
        # 由于 apply 会改变索引结构，我们需要将其恢复为原始结构
        result = result.reset_index(level=0, drop=True).sort_index()
        return result

def COVARIANCE(df1:pd.DataFrame, df2:pd.DataFrame, p:int=5):  
    if isinstance(df2, np.ndarray) and p != len(df2):
        p = len(df2)
        def cov(window):
            return np.cov(window, df2[:len(window)])
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(cov, raw=True))
    else:
        def rolling_cov(group, df2, p):
            # 获取当前分组的 instrument
            instrument = group.name
            # 从 df2 中提取对应的 instrument 数据
            df2_group = df2.xs(instrument, level='instrument')
            # 计算滚动相关性
            return group.rolling(p, min_periods=2).cov(df2_group)

        # 使用 groupby 和 apply 来计算每个 instrument 的滚动相关性
        result = df1.groupby('instrument').apply(lambda x: rolling_cov(x, df2, p))
        # 由于 apply 会改变索引结构，我们需要将其恢复为原始结构
        result = result.reset_index(level=0, drop=True).sort_index()
        return result

@support_numpy
def STD(df:pd.DataFrame, p:int=20):
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())

@support_numpy
def SIGN(df: pd.DataFrame):
    return np.sign(df)

@support_numpy
def SMA(df:pd.DataFrame, m:float=None, n:float=None):
    """
    Y_{i+1} = m/n*X_i + (1 - m/n)*Y_i
    """
        
    if isinstance(m, int) and m >= 1 and n is None:
        return df.groupby('instrument').transform(lambda x: x.rolling(m, min_periods=1).mean())
    else:
        return df.groupby('instrument').transform(lambda x: x.ewm(alpha=n/m).mean())

@support_numpy
def EMA(df:pd.DataFrame, p):
    return df.groupby('instrument').transform(lambda x: x.ewm(span=int(p), min_periods=1).mean())
    
@support_numpy
def WMA(df:pd.DataFrame, p:int=20):
    # 计算权重，最近的数据（i=0）有最大的权重
    weights = [0.9**i for i in range(p)][::-1]
    def calculate_wma(window):
        # if len(window) < p:
        #     w = weights[:len(window)]
        # else:
        #     w = weights
        return (window * weights[:len(window)]).sum() / sum(weights[:len(window)])

    # 应用权重计算滑动WMA
    return df.groupby('instrument').transform(lambda x: x.rolling(window=p, min_periods=1).apply(calculate_wma, raw=True))

@support_numpy
def COUNT(cond:pd.DataFrame, p:int=20):
    return cond.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@support_numpy
def SUMIF(df:pd.DataFrame, p:int, cond:pd.DataFrame):
    return (df * cond).groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@support_numpy
def FILTER(df:pd.DataFrame, cond:pd.DataFrame):
    """
    Filtering A based on condition
    """
    return df.mul(cond)
    

@support_numpy
def PROD(df:pd.DataFrame, p:int=5):
    # 使用rolling方法创建一个滑动窗口，然后应用累乘
    if isinstance(p, int):
        return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lambda x: x.prod(), raw=True))
    else:
        return df.mul(p)    

@support_numpy
def DECAYLINEAR(df:pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"DECAYLINEAR仅接收正整数参数n，接收到{type(p).__name__}")
    decay_weights = np.arange(1, p+1, 1)
    decay_weights = decay_weights / decay_weights.sum()
    
    def calculate_deycaylinear(window):
        return (window * decay_weights[:len(window)]).sum()
    
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(calculate_deycaylinear, raw=True))

@support_numpy
def HIGHDAY(df:pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"HIGHDAY仅接收正整数参数n，接收到{type(p).__name__}")
    def highday(window):
        return len(window) - window.argmax(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(highday, raw=True))

@support_numpy
def LOWDAY(df:pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"LOWDAY仅接收正整数参数n，接收到{type(p).__name__}")
    def lowday(window):
        return len(window) - window.argmin(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lowday, raw=True))
    

def SEQUENCE(n):
    assert isinstance(n, int), ValueError(f"SEQUENCE(n)仅接收正整数参数n，接收到{type(n).__name__}")
    return np.linspace(1, n, n, dtype=np.float32)

@support_numpy
def SUMAC(df:pd.DataFrame, p:int=10):
    assert isinstance(p, int), ValueError(f"SUMAC仅接收正整数参数n，接收到{type(p).__name__}")
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())


def _REGBETA(A:pd.DataFrame, B:pd.DataFrame, p:int=5):
    """
    计算A的每一个列向量回归到向量B的系数
    """
    if len(B.shape) <= 1:
        if len(B.shape) == 0:
            B = np.array([B])
        index = A.columns
        A = np.array(A)[-len(B):].transpose(1, 0)
        B = np.array(B)
        assert len(B.shape) == 1, 'B必须为一个向量'
        if p is not None:
            A = A[:, -p:]
            B = B[-p:]
        
        # 检查A和B的维度是否相同
        if A.shape[1] != B.shape[0]:
            raise ValueError("A 和 B 必须具有相同的维度")

        # 计算回归系数
        A_mean = np.mean(A, axis=1)
        B_mean = np.mean(B, axis=0)
        
        numerator = np.sum((A - np.expand_dims(A_mean, 1)) * (B - B_mean), axis=1)
        denominator = np.sum((A - np.expand_dims(A_mean, 1))**2, axis=1) 
        
        # if (denominator == 0).any():
        #     raise ValueError("无法计算回归系数，因为分母为零")

        beta = numerator / denominator
        
    elif len(B.shape) == 2: 
        index = A.columns
        A = np.array(A).transpose(1, 0)
        B = np.array(B).transpose(1, 0)
        if p is not None:
            A = A[:, -p:]
            B = B[:, -p:]
        # 确保A和B的维度相同
        if A.shape != B.shape:
            raise ValueError("A 和 B 必须具有相同的维度")
        
        # 计算回归系数
        A_mean = np.mean(A, axis=1)
        B_mean = np.mean(B, axis=1)

        numerator = np.sum((A - A_mean[:, np.newaxis]) * (B - B_mean[:, np.newaxis]), axis=1)
        denominator = np.sum((A - A_mean[:, np.newaxis])**2, axis=1)
        
        # if (denominator == 0).any():
        #     raise ValueError("无法计算回归系数，因为分母为零")
        beta = numerator / denominator
    
    return pd.Series(beta, index=index)


def REGBETA(A:pd.DataFrame, B:pd.DataFrame, p:int=5):
    """
    使用滑动窗口计算A和B之间的regression beta值
    """
    assert isinstance(p, int), ValueError(f"REGBETA仅接收正整数参数n，接收到{type(p).__name__}")
    assert isinstance(A, pd.DataFrame), ValueError(f"REGBETA仅接收pd.DataFrame作为A的类型，接收到{type(A).__name__}")
    
    index = A.index
    columns = A.columns
    
    result = []
    for i in range(1, len(A)+1):
        A_window = A.iloc[max(0, i-p):i]
        if i == 0:
            beta = pd.Series(np.zeros_like(A_window))
            result.append(beta)
            continue
        if len(B.shape) == 1:
            B_window = B.iloc[:A_window.shape[0]]
        # 第二维为1的矩阵
        elif isinstance(B, np.ndarray) and len(B.shape) == 2 and B.shape[1] == 1:
            if B.shape[0] == A.shape[0]:
                B_window = B[max(0, i-p):i, 0]
            else:
                B_window = B[:A_window.shape[0], 0]
        # 矩阵
        elif B.shape == A.shape:
            B_window = B.iloc[max(0, i-p):i]
        else:
            raise NotImplementedError("REGBETA()的第二个输入变量的形式不支持")
            
        beta = _REGBETA(A_window, B_window, p)
        result.append(beta)
        
    return pd.DataFrame(result, index=index, columns=columns)


def _REGRESI(A:pd.DataFrame, B:pd.DataFrame, p:int=5):
    if len(B.shape) == 1:
        index = A.columns
        A = np.array(A)[-len(B):].transpose(1, 0)
        B = np.array(B)
        assert len(B.shape) == 1, 'B必须为一个向量'
        if p is not None:
            A = A[:, -p:]
            B = B[-p:]

        # 确保A和B的维度相同
        if A.shape[1] != B.shape[0]:
            raise ValueError("A 和 B 必须具有相同的维度")

        # 计算回归系数
        A_mean = np.mean(A, axis=1)
        B_mean = np.mean(B, axis=0)
        
        numerator = np.sum((A - np.expand_dims(A_mean, 1)) * (B - B_mean), axis=1)
        denominator = np.sum((A - np.expand_dims(A_mean, 1))**2, axis=1)
        
        # if (denominator == 0).any():
        #     raise ValueError("无法计算回归系数，因为分母为零")

        beta = numerator / denominator
        
        # 计算预测值
        B_hat = A * beta[:, np.newaxis]
        # 计算残差
        residuals = np.sum(B - B_hat, axis=1)
    
    # 计算A的每一个列向量回归到矩阵B的每一个列向量的残差
    elif len(B.shape) == 2: 
        index = A.columns
        A = np.array(A).transpose(1, 0)
        B = np.array(B).transpose(1, 0)
        if p is not None:
            A = A[:, -p:]
            B = B[:, -p:]
            
        # 确保A和B的维度相同
        if A.shape != B.shape:
            raise ValueError("A 和 B 必须具有相同的维度")
        
        # 计算回归系数
        A_mean = np.mean(A, axis=1)
        B_mean = np.mean(B, axis=1)
        numerator = np.sum((A - A_mean[:, np.newaxis]) * (B - B_mean[:, np.newaxis]), axis=1)
        denominator = np.sum((A - A_mean[:, np.newaxis])**2, axis=1)
        
        # if (denominator == 0).any():
        #     raise ValueError("无法计算回归系数，因为分母为零")
        
        beta = numerator / denominator
        # 计算预测值
        B_hat = A * beta[:, np.newaxis]
        residuals = np.sum(B - B_hat, axis=1)
        
    return pd.Series(residuals, index=index)


def REGRESI(A:pd.DataFrame, B:pd.DataFrame, p:int=5):
    """
    使用滑动窗口计算A和B之间的regression residual值
    """
    assert isinstance(p, int), ValueError(f"REGRESI仅接收正整数参数n，接收到{type(p).__name__}")
    assert isinstance(A, pd.DataFrame), ValueError(f"REGRESI仅接收pd.DataFrame作为A的类型，接收到{type(A).__name__}")
    index = A.index
    columns = A.columns
    
    result = []
    for i in range(1, len(A)+1):
        A_window = A.iloc[max(0, i-p):i]
        if i == 0:
            beta = pd.Series(np.zeros_like(A_window))
            result.append(beta)
            continue
        
        if len(B.shape) == 1:
            B_window = B.iloc[:A_window.shape[0]]
            
        elif isinstance(B, np.ndarray) and len(B.shape) == 2 and B.shape[1] == 1:
            if B.shape[0] == A.shape[0]:
                B_window = B[max(0, i-p):i, 0]
            else:
                B_window = B[:A_window.shape[0], 0]
        
        elif B.shape == A.shape:
            B_window = B.iloc[max(0, i-p):i]
            
        else:
            raise NotImplementedError("REGRESI()的第二个输入变量的形式不支持")
            
        beta = _REGRESI(A_window, B_window, p)
        result.append(beta)
        
    return pd.DataFrame(result, index=index, columns=columns)
        
        
### 数学运算
@support_numpy
def EXP(df:pd.DataFrame):
    return df.apply(np.exp)

@support_numpy
def SQRT(df: pd.DataFrame):
    return df.apply(np.sqrt)

@support_numpy
def LOG(df:pd.DataFrame):
    return df.apply(np.log)

@support_numpy
def POW(df:pd.DataFrame, n:int):
    return np.power(df, n)

@support_numpy
def TS_ZSCORE(df: pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"TS_ZSCORE仅接收正整数参数n，接收到{type(p).__name__}")
    # assert isinstance(df, pd.DataFrame), ValueError(f"TS_ZSCORE仅接收pd.DataFrame作为A的类型，接收到{type(df).__name__}")
    return (df - df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())) / df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())

@support_numpy
def ZSCORE(df):
    # 在每个时间截面上计算平均值和标准差
    mean = df.groupby('datetime').mean()
    std = df.groupby('datetime').std()
    
    # 计算z-score: (X - μ) / σ
    zscore = (df - mean) / std
    return zscore

def ADD(df1, df2):
    return np.add(df1, df2)
        
        
def SUBTRACT(df1, df2):
    return np.subtract(df1, df2)
    
def MULTIPLY(df1, df2):
    return np.multiply(df1, df2)
    
def DIVIDE(df1, df2):
    return np.divide(df1, df2)
    
def AND(df1, df2):
    return np.bitwise_and(df1.astype(np.bool_), df2.astype(np.bool_))

def OR(df1, df2):
    return np.bitwise_or(df1.astype(np.bool_), df2.astype(np.bool_))


def RSI(price_df, window=14):
    # 计算价格变化
    price_change = DELTA(price_df, 1)
    
    # 分别计算上涨和下跌（使用向量化操作）
    up = (price_change > 0) * price_change
    down = (price_change < 0) * ABS(price_change)
    
    # 计算EMA
    avg_up = EMA(up, window)
    avg_down = EMA(down, window)
    
    # 计算RSI
    rsi = 100 - (100 / (1 + (avg_up / avg_down)))
    return rsi

def BB_MIDDLE(price_df, window=20):
    # 计算中轨(移动平均线)
    middle_band = SMA(price_df, window)
    return middle_band

def BB_UPPER(price_df, window=20):
    # 计算中轨
    middle_band = BB_MIDDLE(price_df, window)
    # 计算标准差
    std = STD(price_df, window)
    # 计算上轨
    upper_band = middle_band + 2 * std
    return upper_band

def BB_LOWER(price_df, window=20):
    # 计算中轨
    middle_band = BB_MIDDLE(price_df, window)
    # 计算标准差
    std = STD(price_df, window)
    # 计算下轨
    lower_band = middle_band - 2 * std
    return lower_band