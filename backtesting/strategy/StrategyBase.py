from datetime import datetime

class StrategyBase:
    def __init__(self, df_historical_data = None):
        self.df_historical_data = df_historical_data

    def generate_trading_signal(data, datetime : datetime= None):
        # Allow trader to define their own trading strategy logic
        pass
    
    # 0 -> hold , 1 -> buy , -1 -> sell