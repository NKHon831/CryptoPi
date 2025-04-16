from backtesting.model import FinalAlphaModel

class BaseStrategy:
    def __init__(self):
        pass

    def generate_trading_signals(self, data):
        pass

    def buy(self):
        return "BUY"
        
    def sell(self):
        return "SELL"


class SampleStrategy(BaseStrategy):
    def __init__(self):
        self.finalAlphaModel = FinalAlphaModel()

    def generate_trading_signals(self, data):
        predicted_signal = self.finalAlphaModel.predict()
        return self.buy() if predicted_signal == 'buy' else self.sell()
        