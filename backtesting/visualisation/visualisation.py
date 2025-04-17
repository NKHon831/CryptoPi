import plotly.graph_objects as go
import pandas as pd
import os

class MarketVisualisation:
  def __init__(self, market_data):
    self.market_data = market_data
    self.charts = {}

  def plot_price_chart(self, title='BTC-USD Price Chart'):
    df = self.market_data.copy()
    
    fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name=title,
    showlegend=True
    )])

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Price (k USD/ BTC)',
      xaxis_rangeslider_visible=True
      )
    
    self.charts["Price Chart"] = fig
    print("Price Chart created and added to charts list.")

    return fig
  
class StrategyVisualisation(MarketVisualisation):
  def __init__(self, performance_data, performance_metrics, market_data=None):
    self.market_data = None
    self.performance_data = performance_data
    self.performance_metrics = performance_metrics
    self.chart_mapping = {
      "Equity Curve": self.plot_equity_curve,
      "Performance Metrics": self.tabulate_metrics,
      "Daily Returns": self.plot_daily_returns,
      "Price Chart with Signals": self.plot_price_with_signals
    }
    self.charts = {}
    if market_data is not None:
      self.market_data = market_data
      super().__init__(market_data)
      self.chart_mapping["Price Chart"] = self.plot_price_chart
  
  def plot_equity_curve(self, title="Equity Curve"):
    df = self.performance_data.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
      x=df['datetime'],
      y=df['equity_value'],
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
    df = self.performance_data.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
      x=df['datetime'],
      y=df['daily_return'],
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

  def plot_price_with_signals(self, title="Price Chart with Signals", simple_plot_only=False):
    fig = go.Figure()

    if hasattr(self, 'market_data') and self.market_data is not None and not simple_plot_only:
      # Use market data if available for more detailed visualisation
      df_market = self.market_data.copy()
      df_market['datetime'] = pd.to_datetime(df_market['datetime'])

      fig.add_trace(go.Candlestick(
        x=df_market['datetime'],
        open=df_market['open'],
        high=df_market['high'],
        low=df_market['low'],
        close=df_market['close'],
        name='OHLC'
      ))
    else:
      df = self.performance_data.copy()
      df['datetime'] = pd.to_datetime(df['datetime'])
      fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
      ))

    df_signals = self.performance_data.copy()

    # Buy signal markers
    buy_df = df_signals[df_signals['buy_signal'] == 1]
    fig.add_trace(go.Scatter(
      x=buy_df['datetime'],
      y=buy_df['price'],
      mode='markers',
      name='Buy Signal',
      marker=dict(symbol='triangle-up', color='green', size=10)
    ))

    # Sell signal markers
    sell_df = df_signals[df_signals['sell_signal'] == 1]
    fig.add_trace(go.Scatter(
      x=sell_df['datetime'],
      y=sell_df['price'],
      mode='markers',
      name='Sell Signal',
      marker=dict(symbol='triangle-down', color='red', size=10)
    ))

    fig.update_layout(
      title=title,
      xaxis_title='Date',
      yaxis_title='Price (k USD/ BTC)',
      template='plotly_white',
      legend=dict(x=0, y=1)
    )

    self.charts["Price Chart with Signals"] = fig
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
        if chart_name in self.charts_mapping:
          self.charts[chart_name] = self.charts_mapping[chart_name]()
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

  def load_market_data(self, market_data):
    if self.market_data is not None:
      print("Market data already loaded. Reloading is not supported.")
      return
    
    self.market_data = market_data
    super().__init__(market_data)
    self.chart_mapping["Price Chart"] = self.plot_price_chart