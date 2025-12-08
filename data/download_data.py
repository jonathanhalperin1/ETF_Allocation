import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import os

def download_etf_data(tickers, start_date, end_date, output_path="data/etf_prices.csv"):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data.to_csv(output_path)

    return data

def download_vix(start_date, end_date, output_path="data/vix.csv"):
    vix = yf.download("^VIX", start=start_date, end=end_date)["Close"]
    vix.to_csv(output_path)
    return vix

def download_yield_curve(start_date, end_date, output_path="data/yield_curve.csv"):

    print("Downloading yield curve data from FRED...")
    try:
        ten_year = web.DataReader("DGS10", "fred", start_date, end_date)
        three_month = web.DataReader("DGS3MO", "fred", start_date, end_date)
        yield_slope = ten_year["DGS10"] - three_month["DGS3MO"]
        yield_slope = yield_slope.to_frame(name="yield_slope")
        yield_slope.to_csv(output_path)
        print(f"Yield curve data saved to {output_path}")
        print(f"Shape: {yield_slope.shape}")
        return yield_slope
    except Exception as e:
        print(f"Error downloading yield curve data: {e}")
        return None

def download_all_data(start_date="2010-01-01", end_date="2024-01-01", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    tickers = ["SPY", "QQQ", "TLT", "GLD", "SHV"]
    etf_data = download_etf_data(tickers, start_date, end_date, 
                                 output_path=f"{data_dir}/etf_prices.csv")
    vix_data = download_vix(start_date, end_date, 
                           output_path=f"{data_dir}/vix.csv")
    yield_data = download_yield_curve(start_date, end_date, 
                                     output_path=f"{data_dir}/yield_curve.csv")
    
    return etf_data, vix_data, yield_data

if __name__ == "__main__":
    download_all_data()

