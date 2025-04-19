from datetime import datetime
from backtesting.models.regime.regime_detection.market_regime_hmm import MarketRegimeHMM as Model

class MarketRegimeHMM:
    def __init__(self, df_historical_data = None):
        self.df_historical_data = df_historical_data
        self.model = Model()
        self.df_market_regime = self.model.predict()

    def generate_trading_signal(self, data, datetime : datetime= None, index = 0):
        print(self.df_market_regime)
        return None
    
    
    # 0 -> hold , 1 -> buy , -1 -> sell