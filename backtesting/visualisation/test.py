# python -m backtesting.visualisation.test

from backtesting.visualisation.visualisation import MarketVisualisation, StrategyVisualisation
from backtesting.visualisation.comparison.comparison_plot import ComparisonPlot
from backtesting.performance.performance_manager import PerformanceManager
from backtesting.visualisation.tools import convert_CSV_df, reformatDates, convert_JSON_df, convert_JSON_dict, convert_MetricsJSON_dict
from backtesting.performance.tools import convert_JSON_list
from backtesting.visualisation.comparison.data_merger import mergePerformanceData, mergePerformanceMetric

if __name__ == '__main__':
  market_df = reformatDates(convert_CSV_df('backtesting/visualisation/__pycache__/example_market_data.csv', 2))
  print(market_df.head())

  # MarketVisualisation(market_df).plot_price_chart().show()

  # performance_df = convert_JSON_df('backtesting/visualisation/__pycache__/example_performance_data.json')
  # print(performance_df.head())
  # metrics_df = convert_JSON_df('backtesting/visualisation/__pycache__/example_performance_metrics.json')
  # print(metrics_df.head())

  # trades = convert_JSON_list("backtesting/performance/__pycache__/example_raw_trades_data.json")
  # performanceManager = PerformanceManager(trades=trades, initial_capital=100, benchmark_curve=None)
  # scalar_metrics, time_series_metrics = performanceManager.get_metrics()
  # strategyVisualiser = StrategyVisualisation(time_series_metrics, scalar_metrics)
  # charts = strategyVisualiser.plot(all=True)
  # for chart_name, fig in charts.items():
  #   fig.show()
  # print(mergePerformanceData(time_series_metrics, time_series_metrics))
  # print(mergePerformanceMetric(scalar_metrics, scalar_metrics))
  
  # strategyVisualiser = StrategyVisualisation(performance_df, metrics_df)
  # strategyVisualiser = StrategyVisualisation(performance_df, metrics_df, market_data=market_df)
  # charts = strategyVisualiser.plot(all=True)
  # for chart_name, fig in charts.items():
  #   fig.show()
  # strategyVisualiser.export(format='html')
  # strategyVisualiser.export(format='json')

  # doesnt work, deleted
  # strategyVisualiser.export(format='png')
  # strategyVisualiser.export(format='jpeg')
  # strategyVisualiser.export(format='svg')
  # strategyVisualiser.export(format='pdf')

  # performances_dict = convert_JSON_dict('backtesting/visualisation/__pycache__/comparison/example_performances_data.json')
  # print(performances_dict)
  # metrics_dict = convert_MetricsJSON_dict('backtesting/visualisation/__pycache__/comparison/example_performances_metrics.json')
  # print(metrics_dict)
  # comparisonVisualiser = ComparisonPlot(performances_dict, performances_metrics=metrics_dict)
  # comparisonVisualiser.plot_comparison_equity_curve().show()
  # comparisonVisualiser.comparison_metrics_table().show()
  # comparisonVisualiser.plot_comparison_daily_returns().show()
  # comparisonVisualiser.plot(all=True)
  # for chart_name, fig in comparisonVisualiser.charts.items():
  #   fig.show()
  # comparisonVisualiser.export(format='html')
  # comparisonVisualiser.export(format='json')

  # implemented but the trading signals look the same for different strats, commented out
  # comparisonVisualiser.plot_comparison_signals().show()