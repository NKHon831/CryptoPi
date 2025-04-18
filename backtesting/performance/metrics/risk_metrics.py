from backtesting.performance.performance_base import PerformanceBase
from backtesting.performance.metrics.time_series_metrics import TimeSeriesMetrics
import numpy as np

class RiskMetrics(PerformanceBase):
  def __init__(self, base: PerformanceBase):
    self.base = base
    self.timeseries = TimeSeriesMetrics(base)

  def sharpe_ratio(self, risk_free_rate=0.0, annualized=True):
    # Formula: (mean(returns) - risk_free_rate) / std(returns)
    daily_returns = self.timeseries.daily_returns()
    if not daily_returns:
      return 0.0

    returns = np.array(list(daily_returns.values()))

    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate

    mean_excess_return = np.mean(excess_returns)
    volatility = self.timeseries.volatility(annualized=False)

    if volatility == 0.0:
      return 0.0

    sharpe = mean_excess_return / volatility

    if annualized:
      sharpe *= np.sqrt(252)

    return sharpe

  def sortino_ratio(self, risk_free_rate=0.0, annualized=True):
    # Formula: (mean(returns) - risk_free_rate) / downside_std
    returns = self.timeseries.daily_returns()

    daily_rf_rate = risk_free_rate / 252
    excess_returns = np.array(list(returns.values())) - daily_rf_rate
    
    downside_returns = excess_returns[excess_returns < 0]
    
    downside_std = downside_returns.std()

    if downside_std == 0:
      return np.nan

    sortino = excess_returns.mean() / downside_std

    if annualized:
      sortino *= np.sqrt(252)
    return sortino

  def calmar_ratio(self):
    # Formula: Calmar Ratio = Annualized Return / Max Drawdown
    cumulative_return = self.timeseries.cumulative_return()
    base = max(0, 1 + cumulative_return)
    
    equity_curve = self.timeseries.equity_curve()
    trading_days = len(equity_curve)
    annualized_return = (base ** (252 / trading_days)) - 1
    
    max_dd = abs(self.timeseries.max_drawdown())

    calmar = annualized_return / max_dd if max_dd != 0 else np.nan
    return calmar

  def calculate_all(self, risk_free_rate=0.0):
    return {
      "sharpe_ratio": self.sharpe_ratio(risk_free_rate, annualized=False),
      "annualized_sharpe_ratio": self.sharpe_ratio(risk_free_rate, annualized=True),
      "sortino_ratio": self.sortino_ratio(risk_free_rate, annualized=False),
      "annualized_sortino_ratio": self.sortino_ratio(risk_free_rate, annualized=True),
      "calmar_ratio": self.calmar_ratio()
    }