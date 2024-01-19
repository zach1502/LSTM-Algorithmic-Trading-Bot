import pandas as pd
import os
import talib
from enum import Enum

def calculate_ema(data, window):
    closed = data['Close']
    return closed.ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(data, window):
    closed = data['Close']
    sma = closed.rolling(window=window).mean()
    std = closed.rolling(window=window).std()
    bollinger_upper = sma + (std * 2)
    bollinger_lower = sma - (std * 2)
    return bollinger_upper, bollinger_lower

def calculate_rsi(data, window):
    closed = data['Close']
    delta = closed.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def calculate_macd(data, window_slow=26, window_fast=12, signal=9):
    ema_fast = calculate_ema(data, window_fast)
    ema_slow = calculate_ema(data, window_slow)
    macd = ema_fast - ema_slow
    signal_line = macd.rolling(window=signal).mean()
    return macd, signal_line

def calculate_OBV(data):
    close = data['Close']
    volume = data['Volume']
    obv = talib.OBV(close, volume)
    return obv

def calculate_ATR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    atr = talib.ATR(high, low, close, timeperiod=window)
    return atr

def calculate_ADX(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    adx = talib.ADX(high, low, close, timeperiod=window)
    return adx

def calculate_stochastic(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    slowk, slowd = talib.STOCH(high, low, close)
    return slowk, slowd

def calculate_AD(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    ad = talib.AD(high, low, close, volume)
    return ad

def calculate_StdDev(data, window):
    closed = data['Close']
    return closed.rolling(window=window).std()

def calculate_LinearReg(data, window):
    closed = data['Close']
    return talib.LINEARREG(closed, timeperiod=window)

def calculate_MFI(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    return talib.MFI(high, low, close, volume, timeperiod=window)

def calculate_MOM(data, window):
    closed = data['Close']
    return talib.MOM(closed, timeperiod=window)

def calculate_ULTOSC(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return talib.ULTOSC(high, low, close)

def calculate_WillR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return talib.WILLR(high, low, close, timeperiod=window)

def calculate_NATR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return talib.NATR(high, low, close, timeperiod=window)

def calculate_TRANGE(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return talib.TRANGE(high, low, close)

def calculate_WCLPRICE(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return talib.WCLPRICE(high, low, close)

def calculate_HT_DCPERIOD(data, _):
    close = data['Close']
    return talib.HT_DCPERIOD(close)

def calculate_BETA(data, window):
    high = data['High']
    low = data['Low']
    return talib.BETA(high, low, timeperiod=window)

def calculate_VAR(data, window):
    close = data['Close']
    return talib.VAR(close, timeperiod=window)

def process_chunk(chunk, overlap, indicators, window_size):
    chunk_combined = pd.concat([overlap, chunk])

    for indicator_name, indicator_func_param in indicators.items():
        indicator_func, params = indicator_func_param

        if indicator_name.startswith('BB_'):
            upper, lower = indicator_func(chunk_combined, params)
            chunk_combined[f'{indicator_name}_upper'] = upper
            chunk_combined[f'{indicator_name}_lower'] = lower
        elif indicator_name.startswith('MACD_'):
            macd, signal = indicator_func(chunk_combined, *params)
            chunk_combined[f'{indicator_name}_macd'] = macd
            chunk_combined[f'{indicator_name}_signal'] = signal
        elif indicator_name == 'OBV':
            chunk_combined[indicator_name] = indicator_func(chunk_combined)
        elif indicator_name.startswith('ATR_') or indicator_name.startswith('ADX_'):
            chunk_combined[indicator_name] = indicator_func(chunk_combined, window_size)
        elif indicator_name == 'Stochastic':
            slowk, slowd = indicator_func(chunk_combined)
            chunk_combined[f'{indicator_name}_slowk'] = slowk
            chunk_combined[f'{indicator_name}_slowd'] = slowd
        else:
            chunk_combined[indicator_name] = indicator_func(chunk_combined, window_size)



    new_overlap = chunk_combined.iloc[-window_size:]
    return chunk_combined.iloc[:-window_size], new_overlap

# Initializations
chunk_size = 10 ** 6
window_size = 50
overlap = pd.DataFrame()

class TimeWindows(Enum):
    super_short = 15
    short = 60
    long = 300

indicators = {
    'EMA_15': (calculate_ema, (TimeWindows.super_short.value)),
    'EMA_60': (calculate_ema, (TimeWindows.short.value)),
    'EMA_300': (calculate_ema, (TimeWindows.long.value)),
    'BB_15': (calculate_bollinger_bands, (TimeWindows.super_short.value)),
    'BB_60': (calculate_bollinger_bands, (TimeWindows.short.value)),
    'BB_300': (calculate_bollinger_bands, (TimeWindows.long.value)),

    # Momentum Indicators
    'RSI_15': (calculate_rsi, (TimeWindows.super_short.value)),
    'RSI_60': (calculate_rsi, (TimeWindows.short.value)),
    'RSI_300': (calculate_rsi, (TimeWindows.long.value)),
    'ULTOSC': (calculate_ULTOSC, ()),

    # Volume Indicators
    'OBV': (calculate_OBV, ()),
    'AD': (calculate_AD, ()),

    # Volatility Indicators
    'ATR_15': (calculate_ATR, (TimeWindows.super_short.value)),
    'ATR_60': (calculate_ATR, (TimeWindows.short.value)),

    # Price Transform
    'WCLPRICE': (calculate_WCLPRICE, ()),

    # Cycle Indicators
    'HT_DCPERIOD': (calculate_HT_DCPERIOD, ()),

    # Statistical Indicators
    'VAR_15': (calculate_VAR, (TimeWindows.super_short.value)),
    'VAR_60': (calculate_VAR, (TimeWindows.short.value)),
    'VAR_300': (calculate_VAR, (TimeWindows.long.value)),

    # Market Strength Indicators
    'MFI_15': (calculate_MFI, (TimeWindows.super_short.value)),
    'MFI_60': (calculate_MFI, (TimeWindows.short.value)),
    'MFI_300': (calculate_MFI, (TimeWindows.long.value)),
}

output_file = './btc_usdt_data/full_btc_usdt_data_feature_engineered.csv'

i=0
for chunk in pd.read_csv('./btc_usdt_data/full_btc_usdt_data_cleaned.csv', chunksize=chunk_size):
    processed_chunk, overlap = process_chunk(chunk, overlap, indicators, window_size)

    processed_chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    i+=len(processed_chunk)
    print(f'Processed chunk, {i} rows saved to {output_file}')
