PPO ETF Portfolio Allocation
--------------------------------

Reinforcement learning portfolio allocation using Proximal Policy Optimization (PPO) for ETF portfolios.

Installation
------------
git clone <repository-url>
cd JohnnyEtf

python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
s
pip install -r requirements.txt

Quick Start (Full Pipeline)
---------------------------
Run data download, feature generation, PPO training, backtesting, and baseline comparison:

python main.py

First run:
1. Download ETF, VIX, and yield curve data (2010–2024)
2. Build features
3. Train the PPO model (rolling-window)
4. Backtest on 2019–2022
5. Run baseline strategies (equal-weight, monthly rebalance)
6. Save metrics and equity curves in `report/`

Outputs
-------
Files created in `report/`:

- strategy_comparison.csv – performance metrics for all strategies
- strategy_equity_curves.csv – portfolio value time series

Data
----
Automatic (default)
Downloaded on first run and stored in `data/`:

- ETF prices: SPY, QQQ, TLT, GLD, SHV (Yahoo Finance)
- VIX (Yahoo Finance)
- Yield curve (FRED)

Manual (optional)
To use your own data, place CSVs in `data/`:

- data/etf_prices.csv
- data/vix.csv
- data/yield_curve.csv

Each file should have a date index.

Project Structure
-----------------
JohnnyEtf/
├── main.py                  # Full pipeline
├── data/
│   ├── download_data.py     # Data download
│   └── *.csv                # Stored data
├── features/
│   └── feature_engineering.py
├── training/
│   └── train_ppo.py         # PPO training script
├── backtest/
│   ├── backtest_agent.py    # Backtest trained model
│   └── evaluate_metrics.py  # Metrics and comparisons
├── env/
│   └── portfolio_env.py     # Gymnasium environment
├── models/                  # Saved PPO models
├── report/                  # Generated results
└── requirements.txt

Individual Components
---------------------
Train PPO only:

python training/train_ppo.py

Backtest an existing model:

python backtest/backtest_agent.py

Requirements
------------
- Python 3.8+
- Packages listed in requirements.txt
- Internet connection for automatic data download

Troubleshooting
---------------
ModuleNotFoundError:
Activate the virtual environment and reinstall dependencies:

source venv/bin/activate
pip install -r requirements.txt

Data download errors:
- Check internet connection
- Confirm Yahoo Finance and FRED are reachable

Training too slow:
- In main.py, reduce n_windows
- Reduce ``timesteps_per_window

Contact
-------
Open an issue in this repository for questions or problems.
