
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
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

def create_env(features_df, etf_tickers, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(features_df)
    
    window_df = features_df.iloc[start_idx:end_idx].copy()
    
    def _make_env():
        return PortfolioEnv(
            features_df=window_df,
            etf_tickers=etf_tickers,
            initial_balance=100000.0,
            alpha=0.1,
            beta=0.2,
            gamma=0.05
        )
    
    return _make_env

def train_ppo_rolling_window(
    features_df,
    etf_tickers,
    window_size=90,
    n_windows=50,
    timesteps_per_window=5000,
    total_timesteps=None,
    model_save_path="models/ppo_portfolio",
    verbose=1
):
    os.makedirs("models", exist_ok=True)
    
    if total_timesteps is None:
        total_timesteps = n_windows * timesteps_per_window
    
    n_samples = len(features_df)
    max_start = max(0, n_samples - window_size)
    
    print("Initializing PPO model...")
    first_env = DummyVecEnv([create_env(features_df, etf_tickers, 0, min(window_size, n_samples))])
    
    try:
        import tensorboard
        tb_log = "./tensorboard_logs/"
    except ImportError:
        tb_log = None
        if verbose > 0:
            print("TensorBoard not installed. Skipping tensorboard logging.")
    
    model = PPO(
        "MlpPolicy",
        first_env,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose,
        tensorboard_log=tb_log
    )
    
    print(f"Training on {n_windows} random windows of {window_size} days each...")
    timesteps_so_far = 0
    
    for i in range(n_windows):
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = min(start_idx + window_size, n_samples)
        
        env = DummyVecEnv([create_env(features_df, etf_tickers, start_idx, end_idx)])
        
        model.set_env(env)
        
        
        try:
            import tqdm
            import rich
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
        
        model.learn(
            total_timesteps=timesteps_per_window,
            reset_num_timesteps=False,
            progress_bar=use_progress_bar
        )
        
        timesteps_so_far += timesteps_per_window
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_windows} windows ({timesteps_so_far}/{total_timesteps} timesteps)")
            model.save(f"{model_save_path}_checkpoint")
    
    model.save(model_save_path)
    print(f"Training complete! Model saved to {model_save_path}")
    
    return model

def train_ppo_full(
    features_df,
    etf_tickers,
    train_start_idx=0,
    train_end_idx=None,
    total_timesteps=50000,
    model_save_path="models/ppo_portfolio",
    verbose=1
):
    os.makedirs("models", exist_ok=True)
    
    if train_end_idx is None:
        train_end_idx = len(features_df)
    
    train_df = features_df.iloc[train_start_idx:train_end_idx].copy()
    
    print(f"Training PPO on {len(train_df)} days of data...")
    
    env = DummyVecEnv([create_env(train_df, etf_tickers)])
    
    
    try:
        import tensorboard
        tb_log = "./tensorboard_logs/"
    except ImportError:
        tb_log = None
        if verbose > 0:
            print("TensorBoard not installed. Skipping tensorboard logging.")
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose,
        tensorboard_log=tb_log
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="ppo_checkpoint"
    )
    
    
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=use_progress_bar
    )
    
    model.save(model_save_path)
    print(f"Training complete! Model saved to {model_save_path}")
    
    return model

if __name__ == "__main__":
    print("Loading data...")
    etf_data, vix_data, yield_curve_data = load_data()
    
    etf_tickers = ["SPY", "QQQ", "TLT", "GLD", "SHV"]
    
    print("Computing features...")
    features_df = compute_all_features(etf_data, vix_data, yield_curve_data)
    print(f"Features shape: {features_df.shape}")
    
    train_end_date = pd.Timestamp("2019-01-01")
    train_mask = features_df.index < train_end_date
    train_features = features_df[train_mask].copy()
    
    print(f"Training period: {train_features.index[0]} to {train_features.index[-1]}")
    print(f"Training samples: {len(train_features)}")
    
    
    model = train_ppo_rolling_window(
        features_df=train_features,
        etf_tickers=etf_tickers,
        window_size=90,
        n_windows=50,
        timesteps_per_window=5000,
        model_save_path="models/ppo_portfolio",
        verbose=1
    )
    
    print("Training complete!")

