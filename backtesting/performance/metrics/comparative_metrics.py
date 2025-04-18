from backtesting.performance.performance_base import PerformanceBase
from backtesting.performance.metrics.time_series_metrics import TimeSeriesMetrics
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class ComparativeMetrics(PerformanceBase):
  def __init__(self, base: PerformanceBase, benchmark_curve: pd.Series):
    self.base = base
    self.benchmark_curve = benchmark_curve

  def _aligned_returns(self):
    strategy_returns = TimeSeriesMetrics(self.base).daily_returns() # From the strategy
    benchmark_returns = self.benchmark_curve.pct_change().dropna()

    df = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    return df
  
  def alpha_beta(self):
    df = self._aligned_returns()
    X = df['benchmark'].values.reshape(-1, 1)
    y = df['strategy'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_[0] * 252   # annualized
    beta = reg.coef_[0][0]
    return alpha, beta
  
  def tracking_error(self):
    df = self._aligned_returns()
    diff = df['strategy'] - df['benchmark']
    return diff.std() * np.sqrt(252)

  def calculate_all(self):
    alpha, beta = self.alpha_beta()
    return {
      "alpha": alpha,
      "beta": beta,
      "tracking_error": self.tracking_error()
    }