# python -m backtesting.visualisation.test

from backtesting.visualisation.visualisation import MarketVisualisation, StrategyVisualisation
from backtesting.visualisation.tools import convert_CSV_df, reformatDates, convert_JSON_df

if __name__ == '__main__':
  market_df = reformatDates(convert_CSV_df('backtesting/visualisation/__pycache__/example_market_data.csv', 2))
  print(market_df.head())

  # MarketVisualisation(market_df).plot_price_chart().show()

  performance_df = convert_JSON_df('backtesting/visualisation/__pycache__/example_performance_data.json')
  print(performance_df.head())
  metrics_df = convert_JSON_df('backtesting/visualisation/__pycache__/example_performance_metrics.json')
  print(metrics_df.head())

  strategyVisualiser = StrategyVisualisation(performance_df, metrics_df)
  # strategyVisualiser = StrategyVisualisation(performance_df, metrics_df, market_data=market_df)
  charts = strategyVisualiser.plot(all=True)
  # for chart_name, fig in charts.items():
  #   fig.show()
  # strategyVisualiser.export(format='html')
  # strategyVisualiser.export(format='json')

  # doesnt work, deleted
  # strategyVisualiser.export(format='png')
  # strategyVisualiser.export(format='jpeg')
  # strategyVisualiser.export(format='svg')
  # strategyVisualiser.export(format='pdf')