# python -m backtesting.performance.test

from backtesting.performance.performance_manager import PerformanceManager
from backtesting.performance.tools import convert_JSON_list

if __name__ == "__main__":
  trades = convert_JSON_list("backtesting/performance/__pycache__/example_raw_trades_data.json")
  print(trades)
  print()
  performanceManager = PerformanceManager(trades=trades, initial_capital=100, benchmark_curve=None)
  scalar_metrics, time_series_metrics = performanceManager.get_metrics()
  print(scalar_metrics)
  print()
  print(time_series_metrics)
  print()