PPO ETF Portfolio Allocation
--------------------------------

Reinforcement learning portfolio allocation using Proximal Policy Optimization (PPO) for ETF portfolios.

Installation
------------
git clone <repository-url>

python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
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
├── project.ipynb            # Jupyter notebook demo
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

Running Individual Components
------------------------------

1. Training the PPO Model
-------------------------
To train the PPO model from scratch:

```bash
python training/train_ppo.py
```

This script will:
- Load data from `data/` directory (downloads if missing)
- Compute features for all available data
- Split data: train on 2010-2018, reserve 2019-2022 for testing
- Train PPO using rolling-window strategy (50 windows of 90 days each)
- Save trained model to `models/ppo_portfolio.zip`

Training parameters can be modified in `training/train_ppo.py`:
- `window_size`: Size of each training window (default: 90 days)
- `n_windows`: Number of random windows to train on (default: 50)
- `timesteps_per_window`: Training steps per window (default: 5000)

Expected time: ~30-60 minutes depending on hardware.

2. Evaluating/Backtesting the Model
------------------------------------
To backtest a trained model:

```bash
python backtest/backtest_agent.py
```

This script will:
- Load the trained model from `models/ppo_portfolio.zip`
- Load data and compute features
- Backtest on the test period (2019-2022 by default)
- Print performance metrics (returns, Sharpe ratio, drawdown, etc.)

**Note**: A trained model must exist first. If `models/ppo_portfolio.zip` doesn't exist, run training first or use `main.py` which handles the full pipeline.

You can modify the backtest period by editing the `start_date` and `end_date` parameters in `backtest/backtest_agent.py`.

3. Running the Jupyter Notebook
---------------------------------
To use the interactive notebook:

```bash
# Make sure Jupyter is installed
pip install jupyter

# Start Jupyter
jupyter notebook project.ipynb
```

Or if using JupyterLab:
```bash
pip install jupyterlab
jupyter lab project.ipynb
```

The notebook (`project.ipynb`) provides:
- Interactive visualization of portfolio performance
- Step-by-step demonstration of the backtesting process
- Comparison of PPO agent vs baseline strategies
- Visual equity curve plots

**Prerequisites for notebook**:
- A trained model must exist at `models/ppo_portfolio.zip`
- Data files must be present in `data/` directory

**Note**: The notebook uses `%matplotlib inline` for inline plots. If you prefer interactive plots, you can change this to `%matplotlib widget` (requires `ipympl`).

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
- In `training/train_ppo.py`, reduce `n_windows` (default: 50)
- Reduce `timesteps_per_window` (default: 5000)
- Reduce `window_size` (default: 90 days)

Notebook won't run:
- Ensure model exists: `models/ppo_portfolio.zip`
- Check that data files exist in `data/` directory
- Verify all imports work: run `python -c "from features.feature_engineering import compute_all_features"`


