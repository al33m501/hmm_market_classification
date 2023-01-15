import numpy as np
import pandas as pd
from empyrical import sortino_ratio
from sklearn.preprocessing import StandardScaler


def figures_to_html(figs, filename="dashboard.html"):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")


def split_preprocess(raw_features, START_CALIBRATION, END_CALIBRATION, scaler=StandardScaler()):
    log_features = np.log1p(raw_features)
    train_idx = log_features[(log_features.index >= START_CALIBRATION) & (log_features.index <= END_CALIBRATION)].index
    train_X = log_features.reindex(train_idx).copy()
    test_idx = log_features[log_features.index > END_CALIBRATION].index
    test_X = log_features.reindex(test_idx).copy()
    scaled_train_X = pd.DataFrame(scaler.fit_transform(train_X)).set_index(train_X.index)
    scaled_train_X.columns = train_X.columns
    scaled_test_X = pd.DataFrame(scaler.transform(test_X)).set_index(test_X.index)
    scaled_test_X.columns = test_X.columns
    return train_X, test_X, scaled_train_X, scaled_test_X


def split_preprocess_wo_log(raw_features, START_CALIBRATION, END_CALIBRATION, scaler=StandardScaler()):
    log_features = raw_features
    train_idx = log_features[(log_features.index >= START_CALIBRATION) & (log_features.index <= END_CALIBRATION)].index
    train_X = log_features.reindex(train_idx).copy()
    test_idx = log_features[log_features.index > END_CALIBRATION].index
    test_X = log_features.reindex(test_idx).copy()
    scaled_train_X = pd.DataFrame(scaler.fit_transform(train_X)).set_index(train_X.index)
    scaled_train_X.columns = train_X.columns
    scaled_test_X = pd.DataFrame(scaler.transform(test_X)).set_index(test_X.index)
    scaled_test_X.columns = test_X.columns
    return train_X.ffill(), test_X.ffill(), scaled_train_X.ffill(), scaled_test_X.ffill()


def cumulative(returns):
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns /= cumulative_returns.iloc[0]
    return cumulative_returns


def regime_return(market_returns):
    regimes_returns = pd.DataFrame()
    for reg in market_returns['state'].unique():
        regimes_returns[reg] = returns_metrics(
            market_returns[market_returns['state'] == reg]['market_returns'].reset_index(drop=True).dropna())
    regimes_returns.index.name = 'state'
    return regimes_returns


def compute_logdiff(series_1, series_2):
    logdiff = np.log(series_1 / series_2).dropna()
    logdiff -= logdiff.iloc[0]
    return logdiff


def cumulative_returns(prices):
    returns = (1 + prices.pct_change().dropna()).cumprod()
    returns /= returns.iloc[0]
    return returns


def returns_metrics(returns):
    return pd.Series({'periods': len(returns),
                      'sortino_ratio': sortino_ratio(returns), })
