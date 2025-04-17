class StrategyBase:
    def __init__(self):
        pass

    def generate_trading_signal(data):
        # Allow trader to define their own trading strategy logic
        pass

    def buy(self):
        #implement buy logic here
        return 1
    
    def hold(self):
        #implement hold logic here
        return 0
    
    def sell(self):
        #implement sell logic here
        return -1
    

    # 0 -> hold , 1 -> buy , -1 -> sell