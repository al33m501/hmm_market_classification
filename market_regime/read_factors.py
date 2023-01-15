import pandas as pd
import pandas_datareader as pdr
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_factors_moex_97():
    bench_name = 'mcftr_price'
    imoex = pdr.get_data_moex("IMOEX", '1990-01-01')[['CLOSE']]
    mcftr = pdr.get_data_moex("MCFTR", '2000-01-01')['CLOSE']
    imoex = imoex[imoex.index <= mcftr.sort_index().index[0]]
    raw_market_data = mcftr.to_frame(f"mcftr_price")
    raw_market_data = pd.concat(
        [(imoex / imoex.iloc[-1, 0] * raw_market_data.iloc[0, 0])['CLOSE'].iloc[:-1], raw_market_data['mcftr_price']])
    return bench_name, raw_market_data.to_frame('mcftr_price')
