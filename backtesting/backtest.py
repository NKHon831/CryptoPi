class Backtest:
    def __init__(self, strategy, dataHandler):
        self.strategy = strategy
        self.dataHandler = dataHandler
        self.results = []

    def run(self):
        for data in self.dataHandler.data:
            signal = self.strategy.generate_trading_signals(data)
            print(f"Generated signal: {signal}")

    def execute_trade(self, signal, row):
        # Placeholder for trade execution logic
        trade_result = {
            'signal': signal,
            'price': row['close'],
            'timestamp': row['timestamp']
        }
        self.results.append(trade_result)

    def get_results(self):
        return self.results