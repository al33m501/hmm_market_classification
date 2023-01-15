import pandas as pd
import numpy as np
import warnings
from empyrical import roll_annual_volatility

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('always', module='hmmlearn')
pd.options.mode.chained_assignment = None


def std_normalized(vals):
    return np.std(vals) / np.mean(vals)


def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]


def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


def pct_change(data, window=1):
    return data.pct_change(window)


def volatility(data, window=2):
    return roll_annual_volatility(data, window)


def ema(data, window=2):
    return data.ewm(span=window, min_periods=0, adjust=False, ignore_na=False).mean()


def custom_feature_generator(factors, column_price, std_period, ma_period, price_deviation_period):
    dataset = factors.copy()
    dataset['mcftr_price(1)_pct_change'] = dataset[column_price].pct_change()
    dataset['std_normalized'] = dataset[column_price].rolling(std_period).apply(std_normalized)
    dataset['ma_ratio'] = dataset[column_price].rolling(ma_period).apply(ma_ratio)
    dataset['price_deviation'] = dataset[column_price].rolling(price_deviation_period).apply(values_deviation)
    dataset = dataset.dropna()
    raw_features = dataset.copy()
    raw_features = raw_features.drop(['mcftr_price'], axis=1).dropna()
    return raw_features
