import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.download_data import download_all_data
from features.feature_engineering import compute_all_features
from training.train_ppo import load_data, train_ppo_rolling_window
from backtest.backtest_agent import backtest_ppo
from backtest.evaluate_metrics import (
    baseline_equal_weight,
    baseline_monthly_rebalance,
    compare_strategies
)


def save_results_to_csv(results_dict, save_dir="report"):
    os.makedirs(save_dir, exist_ok=True)

    comparison_df = compare_strategies(results_dict)
    comparison_path = os.path.join(save_dir, "strategy_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    equity_df = None

    for name, res in results_dict.items():
        dates = res["dates"]
        values = res["portfolio_values"]

        if len(values) == len(dates) + 1:
            values = values[1:]

        if len(values) != len(dates):
            raise ValueError(
                f"Length mismatch for {name}: "
                f"{len(values)} portfolio values vs {len(dates)} dates"
            )

        col_name = name.replace(" ", "_").lower()
        tmp = pd.DataFrame({"date": dates, col_name: values})

        if equity_df is None:
            equity_df = tmp
        else:
            equity_df = pd.merge(equity_df, tmp, on="date", how="outer")

    equity_path = os.path.join(save_dir, "strategy_equity_curves.csv")
    equity_df.to_csv(equity_path, index=False)

    print(f"CSV results saved to {save_dir}/strategy_comparison.csv and {save_dir}/strategy_equity_curves.csv")

def main():
    print("="*60)
    print("PPO ETF Portfolio Allocation - Complete Pipeline")
    print("="*60)
    
    print("\n[Step 1/6] Downloading data...")
    if not os.path.exists("data/etf_prices.csv"):
        download_all_data(start_date="2010-01-01", end_date="2024-01-01")
    else:
        print("Data files already exist. Skipping download.")
    
    print("\n[Step 2/6] Computing features...")
    etf_data, vix_data, yield_curve_data = load_data()
    etf_tickers = ["SPY", "QQQ", "TLT", "GLD", "SHV"]
    features_df = compute_all_features(etf_data, vix_data, yield_curve_data)
    print(f"Features computed. Shape: {features_df.shape}")
    
    print("\n[Step 3/6] Training PPO model...")
    train_end_date = pd.Timestamp("2019-01-01")
    train_mask = features_df.index < train_end_date
    train_features = features_df[train_mask].copy()
    
    model_path = "models/ppo_portfolio"
    if not os.path.exists(model_path + ".zip"):
        model = train_ppo_rolling_window(
            features_df=train_features,
            etf_tickers=etf_tickers,
            window_size=90,
            n_windows=50,
            timesteps_per_window=5000,
            model_save_path=model_path,
            verbose=1
        )
    else:
        print("Trained model already exists. Skipping training.")
    
    print("\n[Step 4/6] Backtesting PPO model...")
    ppo_results = backtest_ppo(
        model_path=model_path,
        features_df=features_df,
        etf_tickers=etf_tickers,
        start_date="2019-01-01",
        end_date="2022-12-31"
    )
    
    print("\n[Step 5/6] Running baseline strategies...")
    test_etf_data = etf_data[
        (etf_data.index >= pd.Timestamp("2019-01-01")) &
        (etf_data.index <= pd.Timestamp("2022-12-31"))
    ]
    
    equal_weight_results = baseline_equal_weight(
        test_etf_data,
        start_date="2019-01-01",
        end_date="2022-12-31"
    )
    
    monthly_rebalance_results = baseline_monthly_rebalance(
        test_etf_data,
        start_date="2019-01-01",
        end_date="2022-12-31"
    )
    
    print("\n[Step 6/6] Saving comparison results to CSV...")
    results_dict = {
        "PPO Agent": ppo_results,
        "Buy and Hold": equal_weight_results,
        "Monthly Rebalance": monthly_rebalance_results
    }
    
    comparison_df = compare_strategies(results_dict)
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    save_results_to_csv(results_dict, save_dir="report")
    
    print("\nPipeline complete! Check 'report/' for CSV results.")

if __name__ == "__main__":
    main()

