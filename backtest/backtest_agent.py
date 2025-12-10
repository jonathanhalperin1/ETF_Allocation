
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.portfolio_env import PortfolioEnv
from features.feature_engineering import compute_all_features

def load_data(data_dir="data"):
    etf_data = pd.read_csv(f"{data_dir}/etf_prices.csv", index_col=0, parse_dates=True)
    vix_data = pd.read_csv(f"{data_dir}/vix.csv", index_col=0, parse_dates=True).iloc[:, 0]
    
    try:
        yield_curve_data = pd.read_csv(f"{data_dir}/yield_curve.csv", index_col=0, parse_dates=True)
    except:
        print("Warning: Could not load yield curve data. Using zeros.")
        yield_curve_data = None
    
    return etf_data, vix_data, yield_curve_data

def backtest_ppo(
    model_path,
    features_df,
    etf_tickers,
    start_date=None,
    end_date=None,
    initial_balance=100000.0
):

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    test_features = features_df.copy()
    if start_date is not None:
        test_features = test_features[test_features.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        test_features = test_features[test_features.index <= pd.Timestamp(end_date)]
    
    print(f"Backtesting on {len(test_features)} days")
    print(f"Period: {test_features.index[0]} to {test_features.index[-1]}")
    
    env = PortfolioEnv(
        features_df=test_features,
        etf_tickers=etf_tickers,
        initial_balance=initial_balance
    )
    
    obs, info = env.reset()
    done = False
    actions_taken = []
    portfolio_values = [initial_balance]
    weights_history = []
    returns_history = []
    
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info["portfolio_value"])
        weights_history.append(info["weights"])
        returns_history.append(info["net_return"])
        
        step += 1
        if step % 100 == 0:
            print(f"Step {step}/{len(features_df)}, Portfolio Value: ${info['portfolio_value']:.2f}")
    
    portfolio_values = np.array(portfolio_values)
    returns_array = np.array(returns_history)
    weights_array = np.array(weights_history)
    
    from backtest.evaluate_metrics import compute_metrics
    
    metrics = compute_metrics(portfolio_values, returns_array, test_features.index)
    
    turnover = np.mean([np.sum(np.abs(weights_array[i] - weights_array[i-1])) 
                       for i in range(1, len(weights_array))])
    
    results = {
        "portfolio_values": portfolio_values,
        "weights_history": weights_array,
        "returns": returns_array,
        "dates": test_features.index,
        "turnover": turnover,
        "final_value": portfolio_values[-1],
        **metrics
    }
    
    return results

if __name__ == "__main__":
    print("Loading data...")
    etf_data, vix_data, yield_curve_data = load_data()
    
    etf_tickers = ["SPY", "QQQ", "TLT", "GLD", "SHV"]
    
    print("Computing features...")
    features_df = compute_all_features(etf_data, vix_data, yield_curve_data)
    
    model_path = "models/ppo_portfolio"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using training/train_ppo.py")
    else:
        results = backtest_ppo(
            model_path=model_path,
            features_df=features_df,
            etf_tickers=etf_tickers,
            start_date="2019-01-01",
            end_date="2022-12-31"
        )
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']*100:.2f}%")
        print(f"Volatility: {results['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Average Turnover: {results['turnover']:.4f}")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print("="*50)

