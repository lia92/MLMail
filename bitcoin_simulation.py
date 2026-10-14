"""
Bitcoin Price Simulation & Golden Cross Trading Strategy

This script simulates 60 days of Bitcoin daily close prices using Geometric Brownian Motion (GBM)
with realistic crypto market parameters. It then computes 7-day and 30-day moving averages
using pandas, executes a Golden Cross / Death Cross algorithmic trading strategy, prints
a daily ledger of transactions and portfolio status, and displays a final performance summary.
"""

import numpy as np
import pandas as pd


def generate_bitcoin_prices(
    days: int = 60,
    initial_price: float = 60000.0,
    annual_drift: float = 0.15,
    annual_volatility: float = 0.65,
    seed: int = 25,
) -> pd.Series:
    """
    Simulates daily Bitcoin close prices using Geometric Brownian Motion (GBM).

    Parameters:
        days (int): Number of daily time steps to simulate (default 60).
        initial_price (float): Starting price of Bitcoin in USD.
        annual_drift (float): Expected annual return rate (mu).
        annual_volatility (float): Expected annual volatility (sigma).
        seed (int): Seed for pseudo-random number generator for reproducibility.

    Returns:
        pd.Series: Simulated daily Bitcoin prices indexed by day (1..days).
    """
    np.random.seed(seed)
    dt = 1 / 365.0  # Daily time step in years

    # Calculate daily drift and daily volatility
    daily_drift = (annual_drift - 0.5 * (annual_volatility ** 2)) * dt
    daily_volatility = annual_volatility * np.sqrt(dt)

    # Generate daily log returns
    random_shocks = np.random.normal(0, 1, days)
    log_returns = daily_drift + daily_volatility * random_shocks

    # Cumulative price path
    price_factors = np.exp(np.cumsum(log_returns))
    prices = initial_price * price_factors

    days_index = pd.RangeIndex(start=1, stop=days + 1, name="Day")
    return pd.Series(prices, index=days_index, name="Price")


def calculate_moving_averages(prices: pd.Series, short_window: int = 7, long_window: int = 30) -> pd.DataFrame:
    """
    Calculates short-term and long-term simple moving averages using pandas.

    Parameters:
        prices (pd.Series): Daily price series.
        short_window (int): Short-term MA window (e.g., 7 days).
        long_window (int): Long-term MA window (e.g., 30 days).

    Returns:
        pd.DataFrame: DataFrame containing Price, MA7, and MA30.
    """
    df = pd.DataFrame({"Price": prices})
    df[f"MA_{short_window}"] = df["Price"].rolling(window=short_window, min_periods=short_window).mean()
    df[f"MA_{long_window}"] = df["Price"].rolling(window=long_window, min_periods=long_window).mean()
    return df


def run_golden_cross_strategy(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    short_window: int = 7,
    long_window: int = 30,
) -> pd.DataFrame:
    """
    Executes the Golden Cross trading strategy over the price and moving average dataframe.

    Strategy Rules:
    - Golden Cross: When Short MA crosses ABOVE Long MA -> Buy max BTC possible with available cash.
    - Death Cross: When Short MA crosses BELOW Long MA -> Sell all BTC holdings for cash.
    - Hold: No crossover signal detected or insufficient data for moving averages.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'Price', f'MA_{short_window}', f'MA_{long_window}'.
        initial_cash (float): Starting capital in USD.
        short_window (int): Short MA period name lookup.
        long_window (int): Long MA period name lookup.

    Returns:
        pd.DataFrame: Daily ledger DataFrame with simulation results.
    """
    ma_short_col = f"MA_{short_window}"
    ma_long_col = f"MA_{long_window}"

    cash = initial_cash
    btc = 0.0

    signals = []
    actions = []
    cash_history = []
    btc_history = []
    portfolio_values = []
    trade_logs = []

    prev_short = None
    prev_long = None

    for day, row in df.iterrows():
        price = row["Price"]
        ma_short = row[ma_short_col]
        ma_long = row[ma_long_col]

        signal = "HOLD"
        action = "HOLD"

        # Check for crossover only when both MAs are available and previous day MAs are available
        if pd.notna(ma_short) and pd.notna(ma_long) and prev_short is not None and prev_long is not None:
            # Golden Cross: short MA moves from <= long MA to > long MA
            if prev_short <= prev_long and ma_short > ma_long:
                signal = "GOLDEN CROSS (BUY)"
                if cash > 0:
                    btc_bought = cash / price
                    trade_logs.append(
                        f"Day {day:02d}: BUY  {btc_bought:.6f} BTC @ ${price:,.2f} | Spent ${cash:,.2f}"
                    )
                    btc += btc_bought
                    cash = 0.0
                    action = f"BUY {btc_bought:.6f} BTC"
                else:
                    action = "SIGNAL BUY (Already fully invested)"

            # Death Cross: short MA moves from >= long MA to < long MA
            elif prev_short >= prev_long and ma_short < ma_long:
                signal = "DEATH CROSS (SELL)"
                if btc > 0:
                    cash_gained = btc * price
                    trade_logs.append(
                        f"Day {day:02d}: SELL {btc:.6f} BTC @ ${price:,.2f} | Received ${cash_gained:,.2f}"
                    )
                    cash += cash_gained
                    action = f"SELL {btc:.6f} BTC"
                    btc = 0.0
                else:
                    action = "SIGNAL SELL (No BTC held)"

        # Update previous MA values if current are valid
        if pd.notna(ma_short) and pd.notna(ma_long):
            prev_short = ma_short
            prev_long = ma_long

        total_value = cash + (btc * price)

        signals.append(signal)
        actions.append(action)
        cash_history.append(cash)
        btc_history.append(btc)
        portfolio_values.append(total_value)

    ledger = df.copy()
    ledger["Signal"] = signals
    ledger["Action"] = actions
    ledger["Cash ($)"] = cash_history
    ledger["BTC"] = btc_history
    ledger["Portfolio Value ($)"] = portfolio_values

    return ledger, trade_logs


def print_simulation_report(
    ledger: pd.DataFrame,
    trade_logs: list,
    initial_cash: float = 10000.0,
    short_window: int = 7,
    long_window: int = 30,
) -> None:
    """
    Prints a formatted daily ledger and summary performance metrics.
    """
    ma_short_col = f"MA_{short_window}"
    ma_long_col = f"MA_{long_window}"

    print("=" * 105)
    print(" " * 35 + "BITCOIN 60-DAY SIMULATION & TRADING LEDGER")
    print("=" * 105)
    print(
        f"{'Day':<5} | {'Price ($)':<11} | {ma_short_col + ' ($)':<11} | {ma_long_col + ' ($)':<11} | "
        f"{'Signal':<20} | {'BTC':<10} | {'Cash ($)':<12} | {'Portfolio ($)':<12}"
    )
    print("-" * 105)

    for day, row in ledger.iterrows():
        ma_s_str = f"{row[ma_short_col]:11.2f}" if pd.notna(row[ma_short_col]) else "        N/A"
        ma_l_str = f"{row[ma_long_col]:11.2f}" if pd.notna(row[ma_long_col]) else "        N/A"
        print(
            f"{day:<5} | ${row['Price']:10.2f} | ${ma_s_str} | ${ma_l_str} | "
            f"{row['Signal']:<20} | {row['BTC']:10.6f} | ${row['Cash ($)']:11.2f} | ${row['Portfolio Value ($)']:11.2f}"
        )

    print("=" * 105)
    print("\n" + "=" * 60)
    print(" " * 18 + "EXECUTED TRADES LOG")
    print("=" * 60)
    if trade_logs:
        for log in trade_logs:
            print(f" - {log}")
    else:
        print(" No trades were triggered during the 60-day period.")
    print("=" * 60)

    # Calculate final performance stats
    start_price = ledger.iloc[0]["Price"]
    end_price = ledger.iloc[-1]["Price"]
    final_portfolio_val = ledger.iloc[-1]["Portfolio Value ($)"]
    strategy_return = ((final_portfolio_val - initial_cash) / initial_cash) * 100.0

    # Buy and Hold Benchmark
    buy_hold_btc = initial_cash / start_price
    buy_hold_final_val = buy_hold_btc * end_price
    buy_hold_return = ((buy_hold_final_val - initial_cash) / initial_cash) * 100.0

    print("\n" + "=" * 60)
    print(" " * 17 + "FINAL PORTFOLIO PERFORMANCE")
    print("=" * 60)
    print(f" Initial Portfolio Value:  ${initial_cash:,.2f}")
    print(f" Starting BTC Price:       ${start_price:,.2f}")
    print(f" Ending BTC Price:         ${end_price:,.2f} ({((end_price - start_price) / start_price) * 100:+.2f}%)")
    print("-" * 60)
    print(f" Final Strategy Cash:      ${ledger.iloc[-1]['Cash ($)']:,.2f}")
    print(f" Final Strategy BTC:       {ledger.iloc[-1]['BTC']:.6f} BTC")
    print(f" Final Portfolio Value:    ${final_portfolio_val:,.2f}")
    print(f" Strategy Total ROI:       {strategy_return:+.2f}%")
    print("-" * 60)
    print(f" Benchmark (Buy & Hold):   ${buy_hold_final_val:,.2f} ({buy_hold_return:+.2f}%)")
    print(f" Strategy vs Benchmark:    {strategy_return - buy_hold_return:+.2f}%")
    print("=" * 60 + "\n")


def main():
    # 1. Simulate 60 days of BTC price data
    prices = generate_bitcoin_prices(days=60, initial_price=60000.0, seed=25)

    # 2. Compute 7-day and 30-day Moving Averages using pandas
    df_ma = calculate_moving_averages(prices, short_window=7, long_window=30)

    # 3. Execute Golden Cross Trading Strategy
    ledger, trade_logs = run_golden_cross_strategy(df_ma, initial_cash=10000.0, short_window=7, long_window=30)

    # 4. Output daily ledger and performance report
    print_simulation_report(ledger, trade_logs, initial_cash=10000.0, short_window=7, long_window=30)


if __name__ == "__main__":
    main()
