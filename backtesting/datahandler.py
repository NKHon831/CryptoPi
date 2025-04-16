class DataHandler:
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def get_next_data(self):
        if self.current_index < len(self.data):
            data_point = self.data[self.current_index]
            self.current_index += 1
            return data_point
        else:
            return None

    def reset(self):
        self.current_index = 0