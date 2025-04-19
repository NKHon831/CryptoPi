import plotly.graph_objects as go
# import plotly.colors as pc
import pandas as pd
import os

class ComparisonPlot:
  def __init__(self, performances_data, performances_metrics):
    self.performances_data = performances_data # dict with strategy names as keys and DataFrames as values
    self.performances_metrics = performances_metrics
    self.chart_mapping = {
      "Comparison Equity Curve": self.plot_comparison_equity_curve,
      "Comparison Performance Metrics": self.comparison_metrics_table,
      "Comparison Daily Returns": self.plot_comparison_daily_returns,
      # "Comparison Trade Signals": self.plot_comparison_signals
    }
    self.charts = {}

  def plot_comparison_equity_curve(self, title="Comparison Equity Curve", plot_overlapping_only=True):
    performances_dict = self.performances_data.copy()
    
    fig = go.Figure()

    for strategy, data in performances_dict.items():
      datetimes = data["datetimes"]
      equity_values = data["equity_values"]
      
      fig.add_trace(go.Scatter(
        x=datetimes,
        y=equity_values,
        mode='lines',
        name=strategy,
        line=dict(width=2)
      ))

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Equity Value (USD)',
      template='plotly_white',
      hovermode='x unified'
    )

    if plot_overlapping_only:
      min_date = min([min(data["datetimes"]) for data in performances_dict.values()])
      max_date = max([max(data["datetimes"]) for data in performances_dict.values()])
      
      # Filter data within the overlapping time range
      for trace in fig['data']:
        trace.x = [x for x in trace.x if min_date <= x <= max_date]
        trace.y = trace.y[:len(trace.x)]
    
    self.charts["Comparison Equity Curve"] = fig
    print("Comparison Equity Curve created and added to charts list.")

    return fig

  def comparison_metrics_table(self, title="Comparison Performance Metrics"):
    strategies = list(self.performances_metrics.keys())

    all_metrics = set()
    for strategy in self.performances_metrics.values():
      all_metrics.update(strategy.keys())

    all_metrics = sorted(list(all_metrics))

    metric_values = {metric: [] for metric in all_metrics}

    for strategy in strategies:
      strategy_data = self.performances_metrics[strategy]
      for metric in all_metrics:
        metric_values[metric].append(strategy_data.get(metric, ""))

    table_data = []
    table_data.append(["Metric"] + strategies)

    for metric in all_metrics:
      table_data.append([metric] + metric_values[metric])

    fig = go.Figure(data=[go.Table(
      header=dict(values=table_data[0], fill_color='paleturquoise', align='center'),
      cells=dict(values=[row for row in zip(*table_data[1:])], fill_color='lavender', align='center')
    )])

    fig.update_layout(
      title=title,
      template='plotly_white'
    )
    
    self.charts["Comparison Performance Metrics"] = fig
    print("Comparison Performance Metrics table created and added to charts list.")

    return fig

  def plot_comparison_daily_returns(self, title="Comparison Daily Returns", plot_overlapping_only=True):
    performances_dict = self.performances_data.copy()
    returns_dict = {}

    for strat_name, data in performances_dict.items():
      datetimes = pd.to_datetime(data['datetimes'])
      daily_returns = data['daily_returns']
      
      if len(datetimes) == len(daily_returns) + 1:
        datetimes = datetimes[1:]
      
      series = pd.Series(daily_returns, index=datetimes, name=strat_name)
      returns_dict[strat_name] = series

    df_returns = pd.concat(returns_dict.values(), axis=1)

    if plot_overlapping_only:
      df_returns = df_returns.dropna()

    fig = go.Figure()
    for strat in df_returns.columns:
      fig.add_trace(go.Scatter(
        x=df_returns.index,
        y=df_returns[strat],
        mode="lines",
        name=strat
      ))
    
    fig.update_layout(
      title=title,
      xaxis_title="Date",
      yaxis_title="Daily Return",
      template="plotly_white"
    )
    
    self.charts["Comparison Daily Returns"] = fig
    print("Comparison Daily Returns chart created and added to charts list.")

    return fig

  # def plot_comparison_signals(self, title="Comparison Price Chart with Signals", plot_overlapping_only=True):
  #   performances_dict = self.performances_data.copy()
  #   price_dict = {}
  #   buy_signals = {}
  #   sell_signals = {}

  #   for strat_name, data in performances_dict.items():
  #     datetimes = pd.to_datetime(data["datetimes"])
  #     prices = pd.Series(data["price_series"], index=datetimes, name=strat_name)
  #     price_dict[strat_name] = prices

  #     buy_idx = [i for i, x in enumerate(data["buy_signals"]) if x == 1]
  #     sell_idx = [i for i, x in enumerate(data["sell_signals"]) if x == 1]

  #     buy_signals[strat_name] = datetimes[buy_idx]
  #     sell_signals[strat_name] = datetimes[sell_idx]

  #   df_prices = pd.concat(price_dict.values(), axis=1)
  #   if plot_overlapping_only:
  #     df_prices = df_prices.dropna(how='any')

  #   fig = go.Figure()

  #   color_palette = pc.qualitative.Set2
  #   strategy_colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(df_prices.columns)}

  #   for strat_name in df_prices.columns:
  #     strat_prices = price_dict[strat_name]
  #     strat_prices = strat_prices.loc[strat_prices.index.isin(df_prices.index)]
  #     color = strategy_colors[strat_name]

  #     fig.add_trace(go.Scatter(
  #       x=strat_prices.index,
  #       y=strat_prices.values,
  #       mode="lines",
  #       name=f"{strat_name} Price",
  #       line=dict(width=2, color=color)
  #     ))

  #     # Buy signals
  #     for dt in buy_signals[strat_name]:
  #       if dt in strat_prices.index:
  #         fig.add_trace(go.Scatter(
  #           x=[dt],
  #           y=[strat_prices.loc[dt]],
  #           mode="markers",
  #           marker=dict(symbol="triangle-up", color=color, size=10),
  #           name=f"Buy ({strat_name})",
  #           legendgroup=f"Buy_{strat_name}",
  #           showlegend=True
  #         ))

  #     # Sell signals
  #     for dt in sell_signals[strat_name]:
  #       if dt in strat_prices.index:
  #         fig.add_trace(go.Scatter(
  #           x=[dt],
  #           y=[strat_prices.loc[dt]],
  #           mode="markers",
  #           marker=dict(symbol="triangle-down", color=color, size=10),
  #           name=f"Sell ({strat_name})",
  #           legendgroup=f"Sell_{strat_name}",
  #           showlegend=True
  #         ))

  #   fig.update_layout(
  #     title=title,
  #     xaxis_title="Date",
  #     yaxis_title="Price",
  #     template="plotly_white",
  #     legend_title="Strategy & Signals"
  #   )
    
  #   self.charts["Comparison Trade Signals"] = fig
  #   print("Comparison Trade Signals chart created and added to charts list.")
    
  #   return fig

  def plot(self, all=True, charts=None):
    if all:
      for chart in self.chart_mapping.keys():
        self.chart_mapping[chart]()
    else:
      for chart in charts:
        if chart in self.chart_mapping:
          self.chart_mapping[chart]()
        else:
          raise ValueError(f"Chart '{chart}' is not supported.")
    return self.charts
  
  def export(self, format="html", all=True, charts=None, export_dir="exports"):
    allowed_formats = {"html", "json"}
    format = format.lower()

    if format not in allowed_formats:
      raise ValueError(f"Invalid format '{format}'. Allowed formats: {', '.join(allowed_formats)}")

    if all:
      required_charts = set(self.chart_mapping.keys())
    else:
      if not charts:
        raise ValueError("You must specify the 'charts' list when 'all' is False.")
      required_charts = set(charts)
    
    for chart_name in required_charts:
      if chart_name not in self.charts:
        if chart_name in self.chart_mapping:
          self.charts[chart_name] = self.chart_mapping[chart_name]()
        else:
          raise ValueError(f"Chart '{chart_name}' is not supported.")

    target_charts = {name: self.charts[name] for name in required_charts}

    os.makedirs(export_dir, exist_ok=True)
    for name, fig in target_charts.items():
      if not hasattr(fig, f"write_{format}"):
        raise ValueError(f"Cannot export chart '{name}' as {format.upper()}.")

      safe_name = name.replace(" ", "_").lower()
      filepath = os.path.join(export_dir, f"{safe_name}.{format}")

      if format == "html":
        fig.write_html(filepath)
      elif format == "json":
        fig.write_json(filepath)

    print(f"✅ Exported {len(target_charts)} comparison chart(s) to '{export_dir}/' as {format.upper()}")