from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        for key, value in os.environ.items():
            setattr(self, key, value)

# Example usage:
# config = Config()
# print(config.NEWSAPI_API_KEY)