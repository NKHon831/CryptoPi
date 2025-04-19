import plotly.graph_objects as go
import pandas as pd
import os
from backtesting.visualisation.tools import match_portfolio_market_data

class Visualisation():
  def __init__(self, performance_data, performance_metrics, market_data):
    self.market_data = market_data
    self.performance_data = performance_data
    self.performance_metrics = performance_metrics
    self.chart_mapping = {
      "Equity Curve": self.plot_equity_curve,
      "Performance Metrics": self.tabulate_metrics,
      "Daily Returns": self.plot_daily_returns,
      "Price Chart": self.plot_price
    }
    self.charts = {}
  
  def plot_equity_curve(self, title="Equity Curve"):
    fig = go.Figure()

    for strategy_name, data in self.performance_data.items():
      datetime = data['datetimes']
      equity_values = data['equity_values']
      fig.add_trace(go.Scatter(
        x=datetime,
        y=equity_values,
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
      ))

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Equity Value (USD)',
      template='plotly_white',
      hovermode='x unified'
    )

    self.charts["Equity Curve"] = fig
    print("Equity Curve created and added to charts list.")
    
    return fig

  def tabulate_metrics(self, title="Performance Metrics Summary"):
    df = self.performance_metrics.copy()

    if isinstance(df, pd.Series):
      df = df.to_frame().T  # Convert Series to single-row DataFrame

    metrics = df.columns.tolist()
    values = df.iloc[0].tolist()

    fig = go.Figure(data=[go.Table(
      header=dict(
        values=["<b>Metric</b>", "<b>Value</b>"],
        fill_color='lightblue',
        align='left',
        font=dict(size=14)
      ),
      cells=dict(
        values=[metrics, values],
        fill_color='white',
        align='left',
        font=dict(size=12)
      )
    )])

    fig.update_layout(title=title)

    self.charts["Performance Metrics"] = fig
    print("Performance Metrics table created and added to charts list.")

    return fig

  def plot_daily_returns(self, title="Daily Returns"):
    fig = go.Figure()

    for strategy_name, data in self.performance_data.items():
      datetime = data['datetimes']
      daily_return = data['daily_returns']
      fig.add_trace(go.Scatter(
        x=datetime,
        y=daily_return,
        mode='lines',
        name='Daily Returns',
        line=dict(color='green')
      ))

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Daily Return',
      template='plotly_white',
      hovermode='x unified'
    )

    self.charts["Daily Returns"] = fig
    print("Daily Returns chart created and added to charts list.")

    return fig

  def plot_price(self, title="Price Chart", with_signals=True):
    df = self.market_data.copy()
    df['datetime'] = df.index
    
    fig = go.Figure(data=[go.Candlestick(
      x=df['datetime'],
      open=df['open'],
      high=df['high'],
      low=df['low'],
      close=df['close'],
      name=title,
      showlegend=True
    )])

    if with_signals:
      trading_signals = df['trading_signal']
      buy_signals = df[trading_signals == 1]
      sell_signals = df[trading_signals == -1]

      fig.add_trace(go.Scatter(
        x=buy_signals['datetime'],
        y=buy_signals['close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=10)
        ))

      fig.add_trace(go.Scatter(
        x=sell_signals['datetime'],
        y=sell_signals['close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=10)
        ))

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Price (k USD/ BTC)',
      xaxis_rangeslider_visible=True
      )

    self.charts["Price Chart"] = fig
    print("Price Chart created and added to charts list.")

    return fig

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

    print(f"✅ Exported {len(target_charts)} chart(s) to '{export_dir}/' as {format.upper()}")