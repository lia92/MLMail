"""
Unit tests for Bitcoin Price Simulation & Golden Cross Trading Strategy.
"""

import unittest
import pandas as pd
import numpy as np

from bitcoin_simulation import (
    generate_bitcoin_prices,
    calculate_moving_averages,
    run_golden_cross_strategy,
    print_simulation_report,
)


class TestBitcoinSimulation(unittest.TestCase):

    def test_generate_bitcoin_prices(self):
        days = 60
        prices = generate_bitcoin_prices(days=days, initial_price=60000.0, seed=42)

        # Verify type, length and index
        self.assertIsInstance(prices, pd.Series)
        self.assertEqual(len(prices), days)
        self.assertEqual(prices.index[0], 1)
        self.assertEqual(prices.index[-1], days)

        # Verify prices are positive real numbers
        self.assertTrue((prices > 0).all())

        # Verify reproducibility with seed
        prices_seed_42 = generate_bitcoin_prices(days=days, initial_price=60000.0, seed=42)
        pd.testing.assert_series_equal(prices, prices_seed_42)

    def test_calculate_moving_averages(self):
        prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], index=range(1, 9))
        df = calculate_moving_averages(prices, short_window=3, long_window=5)

        self.assertIn("Price", df.columns)
        self.assertIn("MA_3", df.columns)
        self.assertIn("MA_5", df.columns)

        # First 2 values for MA_3 should be NaN
        self.assertTrue(pd.isna(df.loc[1, "MA_3"]))
        self.assertTrue(pd.isna(df.loc[2, "MA_3"]))
        # 3rd value for MA_3 should be mean of 10, 20, 30 = 20.0
        self.assertAlmostEqual(df.loc[3, "MA_3"], 20.0)

        # First 4 values for MA_5 should be NaN
        self.assertTrue(pd.isna(df.loc[4, "MA_5"]))
        # 5th value for MA_5 should be mean of 10, 20, 30, 40, 50 = 30.0
        self.assertAlmostEqual(df.loc[5, "MA_5"], 30.0)

    def test_golden_cross_and_death_cross_signals(self):
        # Create synthetic price data to trigger Golden Cross and Death Cross
        # Short window = 2, Long window = 3
        # Prices crafted so MA_2 crosses MA_3
        # Day 1: 10, Day 2: 10, Day 3: 10 -> MA2=[NaN, 10, 10], MA3=[NaN, NaN, 10]
        # Day 4: 10 -> MA2=10, MA3=10
        # Day 5: 30 -> MA2=20, MA3=16.67 (Golden Cross!)
        # Day 6: 30 -> MA2=30, MA3=23.33
        # Day 7: 0  -> MA2=15, MA3=20 (Death Cross!)

        prices = pd.Series([10.0, 10.0, 10.0, 10.0, 30.0, 30.0, 0.0], index=range(1, 8))
        df = calculate_moving_averages(prices, short_window=2, long_window=3)

        initial_cash = 1000.0
        ledger, logs = run_golden_cross_strategy(df, initial_cash=initial_cash, short_window=2, long_window=3)

        # Check ledger columns
        expected_cols = ["Price", "MA_2", "MA_3", "Signal", "Action", "Cash ($)", "BTC", "Portfolio Value ($)"]
        for col in expected_cols:
            self.assertIn(col, ledger.columns)

        # Golden Cross on Day 5
        self.assertEqual(ledger.loc[5, "Signal"], "GOLDEN CROSS (BUY)")
        self.assertAlmostEqual(ledger.loc[5, "Cash ($)"], 0.0)
        self.assertAlmostEqual(ledger.loc[5, "BTC"], 1000.0 / 30.0) # 33.3333 BTC

        # Death Cross on Day 7
        self.assertEqual(ledger.loc[7, "Signal"], "DEATH CROSS (SELL)")
        self.assertAlmostEqual(ledger.loc[7, "Cash ($)"], 0.0) # Sold at price $0
        self.assertAlmostEqual(ledger.loc[7, "BTC"], 0.0)

        self.assertEqual(len(logs), 2)

    def test_print_simulation_report(self):
        prices = generate_bitcoin_prices(days=35, initial_price=50000.0, seed=1)
        df = calculate_moving_averages(prices, short_window=7, long_window=30)
        ledger, logs = run_golden_cross_strategy(df, initial_cash=10000.0, short_window=7, long_window=30)

        # Ensure print_simulation_report executes without error
        try:
            print_simulation_report(ledger, logs, initial_cash=10000.0, short_window=7, long_window=30)
        except Exception as e:
            self.fail(f"print_simulation_report raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
