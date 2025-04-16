class BackTest:
    def __init__(self, dataHandler, strategy):
        self.dataHandler = dataHandler
        self.strategy = strategy    

    def run(self):
        for data in self.dataHandler.get_data():
            signal = self.strategy.generate_trading_signal(data)
            if signal == 'BUY':
                print(f"Buying at {data}")
            elif signal == 'SELL':
                print(f"Selling at {data}")
            else:
                print(f"No action at {data}")