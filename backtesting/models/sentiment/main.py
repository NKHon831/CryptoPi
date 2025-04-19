# python -m backtesting.models.sentiment.main
# execute whole workflow of sentiment analysis model

from .data import news_retrieval, smposts_retrieval
from backtesting.models.sentiment import clean_text
from backtesting.models.sentiment import analyse
import json
import os

def execute_sentiment_analysis_workflow(query, fromTimeFrame, toTimeFrame):
  # Get data from news and social media posts
  news_api = news_retrieval.NewsAPI()
  smposts_api = smposts_retrieval.TweepyAPI()
  news_data = news_api.get_news(query,fromTimeFrame,toTimeFrame)
  smposts_data = smposts_api.get_tweets(query,fromTimeFrame,toTimeFrame)

  # Reformat dates
  news_data = clean_text.reformatDate(news_data)
  smposts_data = clean_text.reformatDate(smposts_data)

  # Clean the data
  news_data = clean_text.cleanNewsText(news_data)
  smposts_data = clean_text.cleanSMPostsText(smposts_data)

  # Merge data
  data = clean_text.mergeCleanedData(news_data, smposts_data)

  print(len(data), "items retrieved and cleaned, ready for sentiment analysis.")
  print(data)

  # Save results to __pycache__ directory
  output_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, 'cleaned_data.json')
  with open(output_file, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  
  print(f"Data saved to {output_file}")

  # Perform sentiment analysis
  texts = [item["text"] for item in data]
  predictions = analyse.classify_sentiment(texts)

  # Merge predictions into the data
  for i, pred in enumerate(predictions):
    data[i]["label"] = pred["label"]
    data[i]["score"] = pred["score"]
  
  print(data)

  # Save results to __pycache__ directory
  output_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, 'sentiment_analysis_results.json')
  with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)
  
  print(f"Analysis results saved to {output_file}")
  print("Sentiment analysis workflow completed.")

if __name__ == "__main__":
  query="bitcoin OR BTC"
  fromTimeFrame="2025-04-15T00:00:00Z"
  toTimeFrame="2025-04-15T23:59:59Z"
  execute_sentiment_analysis_workflow(query, fromTimeFrame, toTimeFrame)