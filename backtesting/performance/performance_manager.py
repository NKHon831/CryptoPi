from backtesting.performance.performance_base import PerformanceBase
from backtesting.performance.metrics.trade_metrics import TradeMetrics
from backtesting.performance.metrics.time_series_metrics import TimeSeriesMetrics
from backtesting.performance.metrics.risk_metrics import RiskMetrics
# from backtesting.performance.metrics.comparative_metrics import ComparativeMetrics
from typing import List, Dict
import pandas as pd

class PerformanceManager:
  def __init__(self, trades: List[Dict], initial_capital: float, benchmark_curve=None):
    self.base = PerformanceBase(trades, initial_capital)
    self.trade_metrics_calculator = TradeMetrics(self.base)
    self.time_series_metrics_calculator = TimeSeriesMetrics(self.base)
    self.risk_metrics_calculator = RiskMetrics(self.base)
    # self.comparative_metrics_calculator = ComparativeMetrics(self.base, benchmark_curve)

  def get_metrics(self):
    scalar_metric, time_series_metric = self.time_series_metrics_calculator.calculate_all()
    self.scalar_metrics = pd.DataFrame([{
      **self.trade_metrics_calculator.calculate_all(),
      **scalar_metric,
      **self.risk_metrics_calculator.calculate_all(),
      # **self.comparative_metrics_calculator.calculate_all()
    }])
    return self.scalar_metrics, time_series_metric