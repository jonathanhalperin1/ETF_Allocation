import pandas as pd
import numpy as np

def compute_momentum(data, window=20):
    return data.pct_change(window)

def compute_volatility(data, window=20):
    returns = data.pct_change()
    return returns.rolling(window).std()

def compute_drawdown(data, window=252):
    rolling_max = data.rolling(window).max()
    drawdown = (data - rolling_max) / rolling_max
    return drawdown

def normalize_series(series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series - mean
    return (series - mean) / std

def compute_all_features(etf_data, vix_data, yield_curve_data, lookback_window=252):
    all_data = pd.DataFrame(index=etf_data.index)
    for ticker in etf_data.columns:
        all_data[ticker] = etf_data[ticker]
    
    momentum = compute_momentum(etf_data, window=20)
    volatility = compute_volatility(etf_data, window=20)
    drawdown = compute_drawdown(etf_data, window=lookback_window)
    
    for ticker in etf_data.columns:
        all_data[f"{ticker}_momentum"] = momentum[ticker]
    
    for ticker in etf_data.columns:
        all_data[f"{ticker}_volatility"] = volatility[ticker]
    
    for ticker in etf_data.columns:
        all_data[f"{ticker}_drawdown"] = drawdown[ticker]
    
    vix_aligned = vix_data.reindex(all_data.index, method='ffill')
    all_data["vix"] = normalize_series(vix_aligned)
    
    if yield_curve_data is not None:
        if isinstance(yield_curve_data, pd.DataFrame):
            yield_series = yield_curve_data.iloc[:, 0]
        else:
            yield_series = yield_curve_data
        
        yield_aligned = yield_series.reindex(all_data.index, method='ffill')
        all_data["yield_slope"] = normalize_series(yield_aligned)
        
        yield_change = yield_aligned.diff(10)
        all_data["yield_trend"] = normalize_series(yield_change)
    else:
        all_data["yield_slope"] = 0.0
        all_data["yield_trend"] = 0.0
    
    returns = etf_data.pct_change()
    for ticker in etf_data.columns:
        all_data[f"{ticker}_return"] = returns[ticker]
    
    all_data = all_data.dropna()
    
    return all_data

def get_feature_names(etf_tickers):
    features = []
    
    for ticker in etf_tickers:
        features.extend([
            f"{ticker}_momentum",
            f"{ticker}_volatility",
            f"{ticker}_drawdown"
        ])
    
    features.extend(["vix", "yield_slope", "yield_trend"])
    
    return features


