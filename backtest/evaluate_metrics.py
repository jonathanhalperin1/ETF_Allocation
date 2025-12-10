import numpy as np
import pandas as pd
from typing import Dict, List

def compute_metrics(portfolio_values: np.ndarray, returns: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    n_days = len(returns)
    annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / n_days) - 1
    
    volatility = np.std(returns) * np.sqrt(252)
    
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    cumulative = portfolio_values / portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    calmar_ratio = abs(annualized_return * 100 / max_drawdown) if max_drawdown != 0 else 0
    
    positive_returns = np.sum(returns > 0) / len(returns) * 100
    
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return * 100,
        "volatility": volatility * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": positive_returns
    }
    
    return metrics

def baseline_equal_weight(etf_data: pd.DataFrame, start_date=None, end_date=None, initial_balance=100000.0) -> Dict:
    if start_date is not None:
        etf_data = etf_data[etf_data.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        etf_data = etf_data[etf_data.index <= pd.Timestamp(end_date)]
    
    n_etfs = len(etf_data.columns)
    weights = np.ones(n_etfs) / n_etfs
    
    returns = etf_data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1).values
    
    portfolio_values = initial_balance * (1 + portfolio_returns).cumprod()
    portfolio_values = np.concatenate([[initial_balance], portfolio_values])
    
    metrics = compute_metrics(portfolio_values, portfolio_returns, returns.index)
    
    return {
        "portfolio_values": portfolio_values,
        "returns": portfolio_returns,
        "dates": returns.index,
        "weights_history": np.tile(weights, (len(returns), 1)),
        **metrics
    }

def baseline_monthly_rebalance(etf_data: pd.DataFrame, start_date=None, end_date=None, 
                               initial_balance=100000.0, transaction_cost=0.001) -> Dict:
    if start_date is not None:
        etf_data = etf_data[etf_data.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        etf_data = etf_data[etf_data.index <= pd.Timestamp(end_date)]
    
    n_etfs = len(etf_data.columns)
    target_weight = 1.0 / n_etfs
    
    returns = etf_data.pct_change().dropna()
    portfolio_values = [initial_balance]
    portfolio_returns = []
    weights_history = []
    
    current_month = None
    portfolio_value = initial_balance
    current_weights = np.ones(n_etfs) / n_etfs
    
    for date, row in returns.iterrows():
        raw_return = (row.values * current_weights).sum()
        
        transaction_cost_amount = 0.0
        if current_month is None or current_month != date.month:
            target_weights = np.ones(n_etfs) / n_etfs
            turnover = np.sum(np.abs(target_weights - current_weights))
            
            transaction_cost_amount = turnover * transaction_cost
            
            current_weights = np.ones(n_etfs) / n_etfs
            current_month = date.month
        
        net_return = raw_return - transaction_cost_amount
        
        portfolio_value *= (1 + net_return)
        
        current_weights = current_weights * (1 + row.values)
        current_weights = current_weights / current_weights.sum()
        
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(net_return)
        weights_history.append(current_weights.copy())
    
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = np.array(portfolio_returns)
    
    metrics = compute_metrics(portfolio_values, portfolio_returns, returns.index)
    
    return {
        "portfolio_values": portfolio_values,
        "returns": portfolio_returns,
        "dates": returns.index,
        "weights_history": np.array(weights_history),
        **metrics
    }

def baseline_risk_parity(etf_data: pd.DataFrame, start_date=None, end_date=None, 
                        initial_balance=100000.0, lookback=20) -> Dict:
    if start_date is not None:
        etf_data = etf_data[etf_data.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        etf_data = etf_data[etf_data.index <= pd.Timestamp(end_date)]
    
    n_etfs = len(etf_data.columns)
    returns = etf_data.pct_change().dropna()
    
    portfolio_values = [initial_balance]
    portfolio_returns = []
    weights_history = []
    
    portfolio_value = initial_balance
    
    for i in range(lookback, len(returns)):
        rolling_returns = returns.iloc[i-lookback:i]
        volatilities = rolling_returns.std().values
        
        volatilities = np.maximum(volatilities, 1e-6)
        
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        daily_return = (returns.iloc[i].values * weights).sum()
        portfolio_value *= (1 + daily_return)
        
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(daily_return)
        weights_history.append(weights.copy())
    
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = np.array(portfolio_returns)
    
    metrics = compute_metrics(portfolio_values, portfolio_returns, returns.index[lookback:])
    
    return {
        "portfolio_values": portfolio_values,
        "returns": portfolio_returns,
        "dates": returns.index[lookback:],
        "weights_history": np.array(weights_history),
        **metrics
    }

def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    comparison = []
    
    for strategy_name, results in results_dict.items():
        comparison.append({
            "Strategy": strategy_name,
            "Total Return (%)": results["total_return"],
            "Annualized Return (%)": results["annualized_return"],
            "Volatility (%)": results["volatility"],
            "Sharpe Ratio": results["sharpe_ratio"],
            "Max Drawdown (%)": results["max_drawdown"],
            "Calmar Ratio": results["calmar_ratio"],
            "Win Rate (%)": results["win_rate"]
        })
    
    return pd.DataFrame(comparison)

