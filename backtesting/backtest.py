class BackTest:
    def __init__(self, dataHandler, strategy):
        self.dataHandler = dataHandler
        self.strategy = strategy    

    def run(self):
        print("\nRunning Backtest...")
        totalSell = 0
        totalBuy = 0
        df_historicalData = self.dataHandler.get_data()

        # print(df_historicalData.head())
        for index, row in df_historicalData.iterrows():
            trading_signal = self.strategy.generate_trading_signal(row)
            if(trading_signal == 1):
                totalBuy += 1
            elif(trading_signal == -1):
                totalSell += 1
        print("Total Data Points: ", len(df_historicalData))
        print("Total Buy Signals: ", totalBuy)
        print("Total Sell Signals: ", totalSell)

