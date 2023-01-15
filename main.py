import pandas as pd
import warnings
import os
import numpy as np
import pomegranate as pg
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from market_regime.label import get_lower_lows, get_higher_highs
from market_regime.feature_engineering import custom_feature_generator
from market_regime.read_factors import get_factors_moex_97
from market_regime.tools import cumulative, regime_return, figures_to_html

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

if __name__ == '__main__':
    st_scaler = StandardScaler()
    BENCH_NAME, factors_raw = get_factors_moex_97()
    print("Data is loaded!")
    END_TRAIN = '2008-01-01'

    factors = factors_raw.resample('W-FRI').last().ffill()
    raw_features = custom_feature_generator(factors,
                                            column_price='mcftr_price',
                                            std_period=5,
                                            ma_period=5,
                                            price_deviation_period=5)

    trend_weekly = cumulative(factors['mcftr_price'].pct_change().dropna()).to_frame('MCFTR')
    higher_highs_i = get_higher_highs(trend_weekly.values)
    higher_highs = trend_weekly.iloc[[i[1] for i in higher_highs_i]].index

    lower_lows_i = get_lower_lows(trend_weekly.values)
    lower_lows = trend_weekly.iloc[[i[1] for i in lower_lows_i]].index
    trend_weekly.loc[higher_highs, 'relmax'] = 'high'
    trend_weekly.loc[lower_lows, 'relmax'] = 'low'
    relmaxes = trend_weekly.bfill().ffill()

    relmaxes['relmax'] = relmaxes['relmax'].replace({'low': "0", 'high': "1"})
    train_idx = raw_features[raw_features.index < END_TRAIN].index
    train_X = raw_features.reindex(train_idx).copy()
    test_idx = raw_features[raw_features.index > END_TRAIN].index
    test_X = raw_features.reindex(test_idx).copy()
    scaled_train_X = pd.DataFrame(st_scaler.fit_transform(train_X)).set_index(train_X.index)
    scaled_train_X.columns = train_X.columns
    scaled_test_X = pd.DataFrame(st_scaler.transform(test_X)).set_index(test_X.index)
    scaled_test_X.columns = test_X.columns
    y = relmaxes['relmax'].shift(-1)
    train_y = y.reindex(scaled_train_X.index)
    # test_y = y.reindex(scaled_test_X.index)
    # test_y = test_y.ffill()
    unique_labels = np.unique(train_y.values).tolist()

    model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution,
                                              n_components=len(unique_labels),
                                              state_names=unique_labels,
                                              X=[scaled_train_X.values.tolist()],
                                              labels=[train_y.values.tolist()],
                                              algorithm='labeled')
    pred = model.predict(scaled_train_X.values.tolist())
    pred_series = pd.Series(pred).replace({1: "1", 0: "0"})

    pred_test = model.predict(scaled_test_X.values.tolist(), algorithm='map')
    pred_series_test = pd.Series(pred_test).replace({1: "1", 0: "0"}).set_axis(scaled_test_X.index)

    test_daily_dates = factors_raw[factors_raw.index > END_TRAIN].index
    # shift predictions
    regimes = pd.Series(index=test_daily_dates)
    for dt in regimes.index:
        regimes.loc[dt] = pred_series_test[pred_series_test.index < dt].iloc[-1]

    market_with_regimes = pd.concat([np.log1p(factors_raw[BENCH_NAME]), regimes.rename('state').astype(str)],
                                    axis=1).dropna()
    market_with_regimes_plot = px.scatter(market_with_regimes.reset_index(), x=market_with_regimes.index,
                                          y=BENCH_NAME,
                                          color='state',
                                          color_discrete_map={'1': 'green', '0': 'red'},
                                          title='[Test] Supervised log-scale')
    market_pct_with_regimes = pd.concat([raw_features[f'{BENCH_NAME}(1)_pct_change'], regimes.rename('state')],
                                        axis=1).dropna()
    market_with_regimes_distplot = px.histogram(market_pct_with_regimes.reset_index(), x=f'{BENCH_NAME}(1)_pct_change',
                                                color='state',
                                                color_discrete_map={'1': 'green', '0': 'red'},
                                                marginal="violin",
                                                hover_data=market_pct_with_regimes.columns,
                                                nbins=350)
    market_returns = pd.concat(
        [factors_raw[BENCH_NAME].pct_change().dropna().rename('market_returns'), regimes.rename('state')],
        axis=1).dropna()
    regimes_returns = regime_return(market_returns)
    print(regimes_returns)

    os.makedirs('output', exist_ok=True)
    figures_to_html(
        [market_with_regimes_plot, market_with_regimes_distplot],
        filename="output/supervised_backtest.html")
